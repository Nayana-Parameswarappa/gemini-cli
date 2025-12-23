/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { AjvJsonSchemaValidator } from '@modelcontextprotocol/sdk/validation/ajv';
import type {
  jsonSchemaValidator,
  JsonSchemaType,
  JsonSchemaValidator,
} from '@modelcontextprotocol/sdk/validation/types.js';
import { MCPOAuthClientProvider } from '../mcp/mcp-oauth-provider.js';
import type { SSEClientTransportOptions } from '@modelcontextprotocol/sdk/client/sse.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import type { StreamableHTTPClientTransportOptions } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type {
  GetPromptResult,
  Prompt,
  ReadResourceResult,
  Resource,
} from '@modelcontextprotocol/sdk/types.js';
import {
  ListResourcesResultSchema,
  ListRootsRequestSchema,
  ReadResourceResultSchema,
  ResourceListChangedNotificationSchema,
  ToolListChangedNotificationSchema,
  type Tool as McpTool,
} from '@modelcontextprotocol/sdk/types.js';
import { parse } from 'shell-quote';
import type { Config, MCPServerConfig } from '../config/config.js';
import { AuthProviderType } from '../config/config.js';
import { GoogleCredentialProvider } from '../mcp/google-auth-provider.js';
import { ServiceAccountImpersonationProvider } from '../mcp/sa-impersonation-provider.js';
import { DiscoveredMCPTool } from './mcp-tool.js';

import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { CallableTool, FunctionCall, Part, Tool } from '@google/genai';
import { basename } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { McpAuthProvider } from '../mcp/auth-provider.js';
import { MCPOAuthTokenStorage } from '../mcp/oauth-token-storage.js';
import type { PromptRegistry } from '../prompts/prompt-registry.js';
import { getErrorMessage, isAuthenticationError } from '../utils/errors.js';
import type {
  Unsubscribe,
  WorkspaceContext,
} from '../utils/workspaceContext.js';
import { exec } from 'node:child_process';
import type {
  OAuthClientInformation,
  OAuthClientMetadata,
} from '@modelcontextprotocol/sdk/shared/auth.js';
import { URL } from 'node:url';
import * as crypto from 'node:crypto';
import type { ToolRegistry } from './tool-registry.js';
import { debugLogger } from '../utils/debugLogger.js';
import type { MessageBus } from '../confirmation-bus/message-bus.js';
import { coreEvents } from '../utils/events.js';
import type { ResourceRegistry } from '../resources/resource-registry.js';
import {
  sanitizeEnvironment,
  type EnvironmentSanitizationConfig,
} from '../services/environmentSanitization.js';

export const MCP_DEFAULT_TIMEOUT_MSEC = 10 * 60 * 1000; // default to 10 minutes
const CALLBACK_PORT = 8090;
const CALLBACK_URL = `http://localhost:${CALLBACK_PORT}/callback`;

export type DiscoveredMCPPrompt = Prompt & {
  serverName: string;
  invoke: (params: Record<string, unknown>) => Promise<GetPromptResult>;
};

/**
 * OAuth authorization response.
 */
export interface OAuthAuthorizationResponse {
  code: string;
  state: string;
}

/**
 * Enum representing the connection status of an MCP server
 */
export enum MCPServerStatus {
  /** Server is disconnected or experiencing errors */
  DISCONNECTED = 'disconnected',
  /** Server is actively disconnecting */
  DISCONNECTING = 'disconnecting',
  /** Server is in the process of connecting */
  CONNECTING = 'connecting',
  /** Server is connected and ready to use */
  CONNECTED = 'connected',
}

/**
 * Enum representing the overall MCP discovery state
 */
export enum MCPDiscoveryState {
  /** Discovery has not started yet */
  NOT_STARTED = 'not_started',
  /** Discovery is currently in progress */
  IN_PROGRESS = 'in_progress',
  /** Discovery has completed (with or without errors) */
  COMPLETED = 'completed',
}

/**
 * A client for a single MCP server.
 *
 * This class is responsible for connecting to, discovering tools from, and
 * managing the state of a single MCP server.
 */
export class McpClient {
  private client: Client | undefined;
  private transport: Transport | undefined;
  private status: MCPServerStatus = MCPServerStatus.DISCONNECTED;
  private isRefreshingTools: boolean = false;
  private pendingToolRefresh: boolean = false;
  private isRefreshingResources: boolean = false;
  private pendingResourceRefresh: boolean = false;

  constructor(
    private readonly serverName: string,
    private readonly serverConfig: MCPServerConfig,
    private readonly toolRegistry: ToolRegistry,
    private readonly promptRegistry: PromptRegistry,
    private readonly resourceRegistry: ResourceRegistry,
    private readonly workspaceContext: WorkspaceContext,
    private readonly cliConfig: Config,
    private readonly debugMode: boolean,
    private readonly onToolsUpdated?: (signal?: AbortSignal) => Promise<void>,
  ) {}

  /**
   * Connects to the MCP server.
   */
  async connect(): Promise<void> {
    if (this.status !== MCPServerStatus.DISCONNECTED) {
      throw new Error(
        `Can only connect when the client is disconnected, current state is ${this.status}`,
      );
    }
    this.updateStatus(MCPServerStatus.CONNECTING);
    try {
      this.client = await connectToMcpServer(
        this.serverName,
        this.serverConfig,
        this.debugMode,
        this.workspaceContext,
        this.cliConfig.sanitizationConfig,
      );

      this.registerNotificationHandlers();

      const originalOnError = this.client.onerror;
      this.client.onerror = (error) => {
        if (this.status !== MCPServerStatus.CONNECTED) {
          return;
        }
        if (originalOnError) originalOnError(error);
        coreEvents.emitFeedback(
          'error',
          `MCP ERROR (${this.serverName})`,
          error,
        );
        this.updateStatus(MCPServerStatus.DISCONNECTED);
      };
      this.updateStatus(MCPServerStatus.CONNECTED);
    } catch (error) {
      this.updateStatus(MCPServerStatus.DISCONNECTED);
      throw error;
    }
  }

  /**
   * Discovers tools and prompts from the MCP server.
   */
  async discover(cliConfig: Config): Promise<void> {
    this.assertConnected();

    const prompts = await this.discoverPrompts();
    const tools = await this.discoverTools(cliConfig);
    const resources = await this.discoverResources();
    this.updateResourceRegistry(resources);

    if (prompts.length === 0 && tools.length === 0 && resources.length === 0) {
      throw new Error('No prompts, tools, or resources found on the server.');
    }

    for (const tool of tools) {
      this.toolRegistry.registerTool(tool);
    }
    this.toolRegistry.sortTools();
  }

  /**
   * Disconnects from the MCP server.
   */
  async disconnect(): Promise<void> {
    if (this.status !== MCPServerStatus.CONNECTED) {
      return;
    }
    this.toolRegistry.removeMcpToolsByServer(this.serverName);
    this.promptRegistry.removePromptsByServer(this.serverName);
    this.resourceRegistry.removeResourcesByServer(this.serverName);
    this.updateStatus(MCPServerStatus.DISCONNECTING);
    const client = this.client;
    this.client = undefined;
    if (this.transport) {
      await this.transport.close();
    }
    if (client) {
      await client.close();
    }
    this.updateStatus(MCPServerStatus.DISCONNECTED);
  }

  /**
   * Returns the current status of the client.
   */
  getStatus(): MCPServerStatus {
    return this.status;
  }

  private updateStatus(status: MCPServerStatus): void {
    this.status = status;
    updateMCPServerStatus(this.serverName, status);
  }

  private assertConnected(): void {
    if (this.status !== MCPServerStatus.CONNECTED) {
      throw new Error(
        `Client is not connected, must connect before interacting with the server. Current state is ${this.status}`,
      );
    }
  }

  private async discoverTools(
    cliConfig: Config,
    options?: { timeout?: number; signal?: AbortSignal },
  ): Promise<DiscoveredMCPTool[]> {
    this.assertConnected();
    return discoverTools(
      this.serverName,
      this.serverConfig,
      this.client!,
      cliConfig,
      this.toolRegistry.getMessageBus(),
      options ?? {
        timeout: this.serverConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC,
      },
    );
  }

  private async discoverPrompts(): Promise<Prompt[]> {
    this.assertConnected();
    return discoverPrompts(this.serverName, this.client!, this.promptRegistry);
  }

  private async discoverResources(): Promise<Resource[]> {
    this.assertConnected();
    return discoverResources(this.serverName, this.client!);
  }

  private updateResourceRegistry(resources: Resource[]): void {
    this.resourceRegistry.setResourcesForServer(this.serverName, resources);
  }

  async readResource(uri: string): Promise<ReadResourceResult> {
    this.assertConnected();
    return this.client!.request(
      {
        method: 'resources/read',
        params: { uri },
      },
      ReadResourceResultSchema,
    );
  }

  /**
   * Registers notification handlers for dynamic updates from the MCP server.
   * This includes handlers for tool list changes and resource list changes.
   */
  private registerNotificationHandlers(): void {
    if (!this.client) {
      return;
    }

    const capabilities = this.client.getServerCapabilities();

    if (capabilities?.tools?.listChanged) {
      debugLogger.log(
        `Server '${this.serverName}' supports tool updates. Listening for changes...`,
      );

      this.client.setNotificationHandler(
        ToolListChangedNotificationSchema,
        async () => {
          debugLogger.log(
            `🔔 Received tool update notification from '${this.serverName}'`,
          );
          await this.refreshTools();
        },
      );
    }

    if (capabilities?.resources?.listChanged) {
      debugLogger.log(
        `Server '${this.serverName}' supports resource updates. Listening for changes...`,
      );

      this.client.setNotificationHandler(
        ResourceListChangedNotificationSchema,
        async () => {
          debugLogger.log(
            `🔔 Received resource update notification from '${this.serverName}'`,
          );
          await this.refreshResources();
        },
      );
    }
  }

  /**
   * Refreshes the resources for this server by re-querying the MCP `resources/list` endpoint.
   *
   * This method implements a **Coalescing Pattern** to handle rapid bursts of notifications
   * (e.g., during server startup or bulk updates) without overwhelming the server or
   * creating race conditions in the ResourceRegistry.
   */
  private async refreshResources(): Promise<void> {
    if (this.isRefreshingResources) {
      debugLogger.log(
        `Resource refresh for '${this.serverName}' is already in progress. Pending update.`,
      );
      this.pendingResourceRefresh = true;
      return;
    }

    this.isRefreshingResources = true;

    try {
      do {
        this.pendingResourceRefresh = false;

        if (this.status !== MCPServerStatus.CONNECTED || !this.client) break;

        const timeoutMs = this.serverConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC;
        const abortController = new AbortController();
        const timeoutId = setTimeout(() => abortController.abort(), timeoutMs);

        let newResources;
        try {
          newResources = await this.discoverResources();
        } catch (err) {
          debugLogger.error(
            `Resource discovery failed during refresh: ${getErrorMessage(err)}`,
          );
          clearTimeout(timeoutId);
          break;
        }

        this.updateResourceRegistry(newResources);

        clearTimeout(timeoutId);

        coreEvents.emitFeedback(
          'info',
          `Resources updated for server: ${this.serverName}`,
        );
      } while (this.pendingResourceRefresh);
    } catch (error) {
      debugLogger.error(
        `Critical error in resource refresh loop for ${this.serverName}: ${getErrorMessage(error)}`,
      );
    } finally {
      this.isRefreshingResources = false;
      this.pendingResourceRefresh = false;
    }
  }

  getServerConfig(): MCPServerConfig {
    return this.serverConfig;
  }

  getInstructions(): string | undefined {
    return this.client?.getInstructions();
  }

  /**
   * Refreshes the tools for this server by re-querying the MCP `tools/list` endpoint.
   *
   * This method implements a **Coalescing Pattern** to handle rapid bursts of notifications
   * (e.g., during server startup or bulk updates) without overwhelming the server or
   * creating race conditions in the global ToolRegistry.
   */
  private async refreshTools(): Promise<void> {
    if (this.isRefreshingTools) {
      debugLogger.log(
        `Tool refresh for '${this.serverName}' is already in progress. Pending update.`,
      );
      this.pendingToolRefresh = true;
      return;
    }

    this.isRefreshingTools = true;

    try {
      do {
        this.pendingToolRefresh = false;

        if (this.status !== MCPServerStatus.CONNECTED || !this.client) break;

        const timeoutMs = this.serverConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC;
        const abortController = new AbortController();
        const timeoutId = setTimeout(() => abortController.abort(), timeoutMs);

        let newTools;
        try {
          newTools = await this.discoverTools(this.cliConfig, {
            signal: abortController.signal,
          });
        } catch (err) {
          debugLogger.error(
            `Discovery failed during refresh: ${getErrorMessage(err)}`,
          );
          clearTimeout(timeoutId);
          break;
        }

        this.toolRegistry.removeMcpToolsByServer(this.serverName);

        for (const tool of newTools) {
          this.toolRegistry.registerTool(tool);
        }
        this.toolRegistry.sortTools();

        if (this.onToolsUpdated) {
          await this.onToolsUpdated(abortController.signal);
        }

        clearTimeout(timeoutId);

        coreEvents.emitFeedback(
          'info',
          `Tools updated for server: ${this.serverName}`,
        );
      } while (this.pendingToolRefresh);
    } catch (error) {
      debugLogger.error(
        `Critical error in refresh loop for ${this.serverName}: ${getErrorMessage(error)}`,
      );
    } finally {
      this.isRefreshingTools = false;
      this.pendingToolRefresh = false;
    }
  }
}

/**
 * Map to track the status of each MCP server within the core package
 */
const serverStatuses: Map<string, MCPServerStatus> = new Map();

/**
 * Track the overall MCP discovery state
 */
let mcpDiscoveryState: MCPDiscoveryState = MCPDiscoveryState.NOT_STARTED;

/**
 * Map to track which MCP servers have been discovered to require OAuth
 */
export const mcpServerRequiresOAuth: Map<string, boolean> = new Map();

/**
 * Cache of OAuth client providers per MCP server.
 * Key: MCP server name
 * Value: MCPOAuthClientProvider instance
 */
export const mcpOAuthClientProviders: Map<string, MCPOAuthClientProvider> =
  new Map();

/**
 * Event listeners for MCP server status changes
 */
type StatusChangeListener = (
  serverName: string,
  status: MCPServerStatus,
) => void;
const statusChangeListeners: StatusChangeListener[] = [];

/**
 * Add a listener for MCP server status changes
 */
export function addMCPStatusChangeListener(
  listener: StatusChangeListener,
): void {
  statusChangeListeners.push(listener);
}

/**
 * Remove a listener for MCP server status changes
 */
export function removeMCPStatusChangeListener(
  listener: StatusChangeListener,
): void {
  const index = statusChangeListeners.indexOf(listener);
  if (index !== -1) {
    statusChangeListeners.splice(index, 1);
  }
}

/**
 * Update the status of an MCP server
 */
export function updateMCPServerStatus(
  serverName: string,
  status: MCPServerStatus,
): void {
  serverStatuses.set(serverName, status);
  // Notify all listeners
  for (const listener of statusChangeListeners) {
    listener(serverName, status);
  }
}

/**
 * Get the current status of an MCP server
 */
export function getMCPServerStatus(serverName: string): MCPServerStatus {
  return serverStatuses.get(serverName) || MCPServerStatus.DISCONNECTED;
}

/**
 * Get all MCP server statuses
 */
export function getAllMCPServerStatuses(): Map<string, MCPServerStatus> {
  return new Map(serverStatuses);
}

/**
 * Get the current MCP discovery state
 */
export function getMCPDiscoveryState(): MCPDiscoveryState {
  return mcpDiscoveryState;
}

/**
 * Create RequestInit for TransportOptions.
 *
 * @param mcpServerConfig The MCP server configuration
 * @param headers Additional headers
 */
function createTransportRequestInit(
  mcpServerConfig: MCPServerConfig,
  headers: Record<string, string>,
): RequestInit {
  return {
    headers: {
      ...mcpServerConfig.headers,
      ...headers,
    },
  };
}

/**
 * Create an AuthProvider for the MCP Transport.
 *
 * @param mcpServerConfig The MCP server configuration
 */
function createAuthProvider(
  mcpServerConfig: MCPServerConfig,
): McpAuthProvider | undefined {
  if (
    mcpServerConfig.authProviderType ===
    AuthProviderType.SERVICE_ACCOUNT_IMPERSONATION
  ) {
    return new ServiceAccountImpersonationProvider(mcpServerConfig);
  }
  if (
    mcpServerConfig.authProviderType === AuthProviderType.GOOGLE_CREDENTIALS
  ) {
    return new GoogleCredentialProvider(mcpServerConfig);
  }
  return undefined;
}

/**
 * Discovers tools from all configured MCP servers and registers them with the tool registry.
 * It orchestrates the connection and discovery process for each server defined in the
 * configuration, as well as any server specified via a command-line argument.
 *
 * @param mcpServers A record of named MCP server configurations.
 * @param mcpServerCommand An optional command string for a dynamically specified MCP server.
 * @param toolRegistry The central registry where discovered tools will be registered.
 * @returns A promise that resolves when the discovery process has been attempted for all servers.
 */

export async function discoverMcpTools(
  mcpServers: Record<string, MCPServerConfig>,
  mcpServerCommand: string | undefined,
  toolRegistry: ToolRegistry,
  promptRegistry: PromptRegistry,
  debugMode: boolean,
  workspaceContext: WorkspaceContext,
  cliConfig: Config,
): Promise<void> {
  mcpDiscoveryState = MCPDiscoveryState.IN_PROGRESS;
  try {
    mcpServers = populateMcpServerCommand(mcpServers, mcpServerCommand);

    const discoveryPromises = Object.entries(mcpServers).map(
      ([mcpServerName, mcpServerConfig]) =>
        connectAndDiscover(
          mcpServerName,
          mcpServerConfig,
          toolRegistry,
          promptRegistry,
          debugMode,
          workspaceContext,
          cliConfig,
        ),
    );
    await Promise.all(discoveryPromises);
  } finally {
    mcpDiscoveryState = MCPDiscoveryState.COMPLETED;
  }
}

/**
 * A tolerant JSON Schema validator for MCP tool output schemas.
 *
 * Some MCP servers (e.g. third‑party extensions) return complex schemas that
 * include `$defs` / `$ref` chains which can occasionally trip AJV's resolver,
 * causing discovery to fail. This wrapper keeps the default AJV validator for
 * normal operation but falls back to a no‑op validator any time schema
 * compilation throws, so we can still list and use the tool while emitting a
 * debug log.
 */
class LenientJsonSchemaValidator implements jsonSchemaValidator {
  private readonly ajvValidator = new AjvJsonSchemaValidator();

  getValidator<T>(schema: JsonSchemaType): JsonSchemaValidator<T> {
    try {
      return this.ajvValidator.getValidator<T>(schema);
    } catch (error) {
      debugLogger.warn(
        `Failed to compile MCP tool output schema (${
          (schema as Record<string, unknown>)?.['$id'] ?? '<no $id>'
        }): ${error instanceof Error ? error.message : String(error)}. ` +
          'Skipping output validation for this tool.',
      );
      return (input: unknown) => ({
        valid: true as const,
        data: input as T,
        errorMessage: undefined,
      });
    }
  }
}

/** Visible for Testing */
export function populateMcpServerCommand(
  mcpServers: Record<string, MCPServerConfig>,
  mcpServerCommand: string | undefined,
): Record<string, MCPServerConfig> {
  if (mcpServerCommand) {
    const cmd = mcpServerCommand;
    const args = parse(cmd, process.env) as string[];
    if (args.some((arg) => typeof arg !== 'string')) {
      throw new Error('failed to parse mcpServerCommand: ' + cmd);
    }
    // use generic server name 'mcp'
    mcpServers['mcp'] = {
      command: args[0],
      args: args.slice(1),
    };
  }
  return mcpServers;
}

/**
 * Connects to an MCP server and discovers available tools, registering them with the tool registry.
 * This function handles the complete lifecycle of connecting to a server, discovering tools,
 * and cleaning up resources if no tools are found.
 *
 * @param mcpServerName The name identifier for this MCP server
 * @param mcpServerConfig Configuration object containing connection details
 * @param toolRegistry The registry to register discovered tools with
 * @returns Promise that resolves when discovery is complete
 */
export async function connectAndDiscover(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  toolRegistry: ToolRegistry,
  promptRegistry: PromptRegistry,
  debugMode: boolean,
  workspaceContext: WorkspaceContext,
  cliConfig: Config,
): Promise<void> {
  updateMCPServerStatus(mcpServerName, MCPServerStatus.CONNECTING);

  let mcpClient: Client | undefined;
  try {
    mcpClient = await connectToMcpServer(
      mcpServerName,
      mcpServerConfig,
      debugMode,
      workspaceContext,
      cliConfig.sanitizationConfig,
    );

    mcpClient.onerror = (error) => {
      coreEvents.emitFeedback('error', `MCP ERROR (${mcpServerName}):`, error);
      updateMCPServerStatus(mcpServerName, MCPServerStatus.DISCONNECTED);
    };

    // Attempt to discover both prompts and tools
    const prompts = await discoverPrompts(
      mcpServerName,
      mcpClient,
      promptRegistry,
    );
    const tools = await discoverTools(
      mcpServerName,
      mcpServerConfig,
      mcpClient,
      cliConfig,
      toolRegistry.getMessageBus(),
      { timeout: mcpServerConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC },
    );

    // If we have neither prompts nor tools, it's a failed discovery
    if (prompts.length === 0 && tools.length === 0) {
      throw new Error('No prompts or tools found on the server.');
    }

    // If we found anything, the server is connected
    updateMCPServerStatus(mcpServerName, MCPServerStatus.CONNECTED);

    // Register any discovered tools
    for (const tool of tools) {
      toolRegistry.registerTool(tool);
    }
    toolRegistry.sortTools();
  } catch (error) {
    if (mcpClient) {
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      mcpClient.close();
    }
    coreEvents.emitFeedback(
      'error',
      `Error connecting to MCP server '${mcpServerName}': ${getErrorMessage(
        error,
      )}`,
      error,
    );
    updateMCPServerStatus(mcpServerName, MCPServerStatus.DISCONNECTED);
  }
}

/**
 * Discovers and sanitizes tools from a connected MCP client.
 * It retrieves function declarations from the client, filters out disabled tools,
 * generates valid names for them, and wraps them in `DiscoveredMCPTool` instances.
 *
 * @param mcpServerName The name of the MCP server.
 * @param mcpServerConfig The configuration for the MCP server.
 * @param mcpClient The active MCP client instance.
 * @param cliConfig The CLI configuration object.
 * @param messageBus Optional message bus for policy engine integration.
 * @returns A promise that resolves to an array of discovered and enabled tools.
 * @throws An error if no enabled tools are found or if the server provides invalid function declarations.
 */
export async function discoverTools(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  mcpClient: Client,
  cliConfig: Config,
  messageBus?: MessageBus,
  options?: { timeout?: number; signal?: AbortSignal },
): Promise<DiscoveredMCPTool[]> {
  try {
    // Only request tools if the server supports them.
    if (mcpClient.getServerCapabilities()?.tools == null) return [];

    const response = await mcpClient.listTools({}, options);
    const discoveredTools: DiscoveredMCPTool[] = [];
    for (const toolDef of response.tools) {
      try {
        if (!isEnabled(toolDef, mcpServerName, mcpServerConfig)) {
          continue;
        }

        const mcpCallableTool = new McpCallableTool(
          mcpClient,
          toolDef,
          mcpServerConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC,
        );

        const tool = new DiscoveredMCPTool(
          mcpCallableTool,
          mcpServerName,
          toolDef.name,
          toolDef.description ?? '',
          toolDef.inputSchema ?? { type: 'object', properties: {} },
          mcpServerConfig.trust,
          undefined,
          cliConfig,
          mcpServerConfig.extension?.name,
          mcpServerConfig.extension?.id,
          messageBus,
        );

        discoveredTools.push(tool);
      } catch (error) {
        coreEvents.emitFeedback(
          'error',
          `Error discovering tool: '${
            toolDef.name
          }' from MCP server '${mcpServerName}': ${(error as Error).message}`,
          error,
        );
      }
    }
    return discoveredTools;
  } catch (error) {
    if (
      error instanceof Error &&
      !error.message?.includes('Method not found')
    ) {
      coreEvents.emitFeedback(
        'error',
        `Error discovering tools from ${mcpServerName}: ${getErrorMessage(
          error,
        )}`,
        error,
      );
    }
    return [];
  }
}

class McpCallableTool implements CallableTool {
  constructor(
    private readonly client: Client,
    private readonly toolDef: McpTool,
    private readonly timeout: number,
  ) {}

  async tool(): Promise<Tool> {
    return {
      functionDeclarations: [
        {
          name: this.toolDef.name,
          description: this.toolDef.description,
          parametersJsonSchema: this.toolDef.inputSchema,
        },
      ],
    };
  }

  async callTool(functionCalls: FunctionCall[]): Promise<Part[]> {
    // We only expect one function call at a time for MCP tools in this context
    if (functionCalls.length !== 1) {
      throw new Error('McpCallableTool only supports single function call');
    }
    const call = functionCalls[0];

    try {
      const result = await this.client.callTool(
        {
          name: call.name!,
          arguments: call.args as Record<string, unknown>,
        },
        undefined,
        { timeout: this.timeout },
      );

      return [
        {
          functionResponse: {
            name: call.name,
            response: result,
          },
        },
      ];
    } catch (error) {
      // Return error in the format expected by DiscoveredMCPTool
      return [
        {
          functionResponse: {
            name: call.name,
            response: {
              error: {
                message: error instanceof Error ? error.message : String(error),
                isError: true,
              },
            },
          },
        },
      ];
    }
  }
}

/**
 * Discovers and logs prompts from a connected MCP client.
 * It retrieves prompt declarations from the client and logs their names.
 *
 * @param mcpServerName The name of the MCP server.
 * @param mcpClient The active MCP client instance.
 */
export async function discoverPrompts(
  mcpServerName: string,
  mcpClient: Client,
  promptRegistry: PromptRegistry,
): Promise<Prompt[]> {
  try {
    // Only request prompts if the server supports them.
    if (mcpClient.getServerCapabilities()?.prompts == null) return [];

    const response = await mcpClient.listPrompts({});

    for (const prompt of response.prompts) {
      promptRegistry.registerPrompt({
        ...prompt,
        serverName: mcpServerName,
        invoke: (params: Record<string, unknown>) =>
          invokeMcpPrompt(mcpServerName, mcpClient, prompt.name, params),
      });
    }
    return response.prompts;
  } catch (error) {
    // It's okay if this fails, not all servers will have prompts.
    // Don't log an error if the method is not found, which is a common case.
    if (
      error instanceof Error &&
      !error.message?.includes('Method not found')
    ) {
      coreEvents.emitFeedback(
        'error',
        `Error discovering prompts from ${mcpServerName}: ${getErrorMessage(
          error,
        )}`,
        error,
      );
    }
    return [];
  }
}

export async function discoverResources(
  mcpServerName: string,
  mcpClient: Client,
): Promise<Resource[]> {
  if (mcpClient.getServerCapabilities()?.resources == null) {
    return [];
  }

  const resources = await listResources(mcpServerName, mcpClient);
  return resources;
}

async function listResources(
  mcpServerName: string,
  mcpClient: Client,
): Promise<Resource[]> {
  const resources: Resource[] = [];
  let cursor: string | undefined;
  try {
    do {
      const response = await mcpClient.request(
        {
          method: 'resources/list',
          params: cursor ? { cursor } : {},
        },
        ListResourcesResultSchema,
      );
      resources.push(...(response.resources ?? []));
      cursor = response.nextCursor ?? undefined;
    } while (cursor);
  } catch (error) {
    if (error instanceof Error && error.message?.includes('Method not found')) {
      return [];
    }
    coreEvents.emitFeedback(
      'error',
      `Error discovering resources from ${mcpServerName}: ${getErrorMessage(
        error,
      )}`,
      error,
    );
    throw error;
  }
  return resources;
}

/**
 * Invokes a prompt on a connected MCP client.
 *
 * @param mcpServerName The name of the MCP server.
 * @param mcpClient The active MCP client instance.
 * @param promptName The name of the prompt to invoke.
 * @param promptParams The parameters to pass to the prompt.
 * @returns A promise that resolves to the result of the prompt invocation.
 */
export async function invokeMcpPrompt(
  mcpServerName: string,
  mcpClient: Client,
  promptName: string,
  promptParams: Record<string, unknown>,
): Promise<GetPromptResult> {
  try {
    const sanitizedParams: Record<string, string> = {};
    for (const [key, value] of Object.entries(promptParams)) {
      if (value !== undefined && value !== null) {
        sanitizedParams[key] = String(value);
      }
    }

    const response = await mcpClient.getPrompt({
      name: promptName,
      arguments: sanitizedParams,
    });

    return response;
  } catch (error) {
    if (
      error instanceof Error &&
      !error.message?.includes('Method not found')
    ) {
      coreEvents.emitFeedback(
        'error',
        `Error invoking prompt '${promptName}' from ${mcpServerName} ${promptParams}: ${getErrorMessage(
          error,
        )}`,
        error,
      );
    }
    throw error;
  }
}

/**
 * @visiblefortesting
 * Checks if the MCP server configuration has a network transport URL (SSE or HTTP).
 * @param config The MCP server configuration.
 * @returns True if a `url` or `httpUrl` is present, false otherwise.
 */
export function hasNetworkTransport(config: MCPServerConfig): boolean {
  return !!(config.url || config.httpUrl);
}

/**
 * Helper function to get a stored OAuth token for a server.
 * This is used when attempting SSE fallback with OAuth authentication.
 * Note: This returns the stored token without refreshing it if expired.
 *
 * @param serverName The name of the MCP server
 * @returns The access token if stored and not expired, or null otherwise
 */
async function getStoredOAuthToken(serverName: string): Promise<string | null> {
  const tokenStorage = new MCPOAuthTokenStorage();
  const credentials = await tokenStorage.getCredentials(serverName);
  if (!credentials || !credentials.token) return null;

  // Check if token is expired
  if (tokenStorage.isTokenExpired(credentials.token)) {
    return null;
  }

  return credentials.token.accessToken;
}

/**
 * Helper function to create an SSE transport with optional OAuth authentication.
 *
 * @param config The MCP server configuration
 * @param accessToken Optional OAuth access token for authentication
 * @returns A configured SSE transport ready for connection
 */
function createSSETransportWithAuth(
  config: MCPServerConfig,
  accessToken?: string | null,
): SSEClientTransport {
  const headers = {
    ...config.headers,
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
  };

  const options: SSEClientTransportOptions = {};
  if (Object.keys(headers).length > 0) {
    options.requestInit = { headers };
  }

  return new SSEClientTransport(new URL(config.url!), options);
}

/**
 * Helper function to connect a client using SSE transport with optional OAuth.
 *
 * @param client The MCP client to connect
 * @param config The MCP server configuration
 * @param accessToken Optional OAuth access token for authentication
 */
async function connectWithSSETransport(
  client: Client,
  config: MCPServerConfig,
  accessToken?: string | null,
): Promise<void> {
  const transport = createSSETransportWithAuth(config, accessToken);
  await client.connect(transport, {
    timeout: config.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC,
  });
}

/**
 * Creates and connects an MCP client to a server based on the provided configuration.
 * It determines the appropriate transport (Stdio, SSE, or Streamable HTTP) and
 * establishes a connection. It also applies a patch to handle request timeouts.
 *
 * @param mcpServerName The name of the MCP server, used for logging and identification.
 * @param mcpServerConfig The configuration specifying how to connect to the server.
 * @returns A promise that resolves to a connected MCP `Client` instance.
 * @throws An error if the connection fails or the configuration is invalid.
 */
export async function connectToMcpServer(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  debugMode: boolean,
  workspaceContext: WorkspaceContext,
  sanitizationConfig: EnvironmentSanitizationConfig,
): Promise<Client> {
  const mcpClient = new Client(
    {
      name: 'gemini-cli-mcp-client',
      version: '0.0.1',
    },
    {
      // Use a tolerant validator so bad output schemas don't block discovery.
      jsonSchemaValidator: new LenientJsonSchemaValidator(),
    },
  );

  mcpClient.registerCapabilities({
    roots: {
      listChanged: true,
    },
  });

  mcpClient.setRequestHandler(ListRootsRequestSchema, async () => {
    const roots = [];
    for (const dir of workspaceContext.getDirectories()) {
      roots.push({
        uri: pathToFileURL(dir).toString(),
        name: basename(dir),
      });
    }
    return {
      roots,
    };
  });

  let unlistenDirectories: Unsubscribe | undefined =
    workspaceContext.onDirectoriesChanged(async () => {
      try {
        await mcpClient.notification({
          method: 'notifications/roots/list_changed',
        });
      } catch (_) {
        // If this fails, its almost certainly because the connection was closed
        // and we should just stop listening for future directory changes.
        unlistenDirectories?.();
        unlistenDirectories = undefined;
      }
    });

  // Attempt to pro-actively unsubscribe if the mcp client closes. This API is
  // very brittle though so we don't have any guarantees, hence the try/catch
  // above as well.
  //
  // Be a good steward and don't just bash over onclose.
  const oldOnClose = mcpClient.onclose;
  mcpClient.onclose = () => {
    oldOnClose?.();
    unlistenDirectories?.();
    unlistenDirectories = undefined;
  };

  let firstAttemptError: Error | null = null;
  let httpReturned404 = false; // Track if HTTP returned 404 to skip it in OAuth retry
  let sseError: Error | null = null; // Track SSE fallback error

  try {
    const transport = await createTransport(
      mcpServerName,
      mcpServerConfig,
      debugMode,
      sanitizationConfig,
    );
    try {
      await mcpClient.connect(transport, {
        timeout: mcpServerConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC,
      });
      return mcpClient;
    } catch (error) {
      firstAttemptError =
        error instanceof Error ? error : new Error(String(error));
      if (
        isAuthenticationError(error) &&
        hasNetworkTransport(mcpServerConfig)
      ) {
        mcpServerRequiresOAuth.set(mcpServerName, true);
        const callbackPromise = waitForOAuthCallback(generateStateParam());
        const authCode = await callbackPromise;

        if (
          transport instanceof StreamableHTTPClientTransport &&
          typeof transport.finishAuth === 'function'
        ) {
          // Complete the OAuth flow with the authorization code
          await transport.finishAuth(authCode);

          // Close the old transport after finishAuth completes
          try {
            await transport.close();
          } catch {
            throw error;
          }

          // Create a fresh transport - the auth provider now has valid tokens
          const newTransport = await createTransport(
            mcpServerName,
            mcpServerConfig,
            debugMode,
            sanitizationConfig,
          );

          await mcpClient.connect(newTransport, {
            timeout: mcpServerConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC,
          });
          return mcpClient;
        } else {
          throw new Error('Transport does not support finishAuth method');
        }
      } else {
        console.error('❌ Connection failed with non-auth error:', error);
        if (transport) {
          try {
            await transport.close();
          } catch {
            throw error;
          }
        }
        throw error;
      }
    }
  } catch (initialError) {
    let error = initialError;

    if (
      // If not 401, and HTTP failed with url without explicit type, try SSE fallback
      firstAttemptError &&
      mcpServerConfig.url &&
      !mcpServerConfig.type &&
      !mcpServerConfig.httpUrl
    ) {
      // Check if HTTP returned 404 - if so, we know it's not an HTTP server
      httpReturned404 = String(firstAttemptError).includes('404');

      const logMessage = httpReturned404
        ? `HTTP returned 404, trying SSE transport...`
        : `HTTP connection failed, attempting SSE fallback...`;
      debugLogger.log(`MCP server '${mcpServerName}': ${logMessage}`);

      try {
        // Try SSE with stored OAuth token if available
        // This ensures that SSE fallback works for authenticated servers
        await connectWithSSETransport(
          mcpClient,
          mcpServerConfig,
          await getStoredOAuthToken(mcpServerName),
        );

        debugLogger.log(
          `MCP server '${mcpServerName}': Successfully connected using SSE transport.`,
        );
        return mcpClient;
      } catch (sseFallbackError) {
        sseError = sseFallbackError as Error;

        // If SSE also returned 401, handle OAuth below
        if (isAuthenticationError(sseError)) {
          debugLogger.log(
            `MCP server '${mcpServerName}': SSE returned 401, OAuth authentication required.`,
          );
          // Update error to be the SSE error for OAuth handling
          error = sseError;
          // Continue to OAuth handling below
        } else {
          debugLogger.log(
            `MCP server '${mcpServerName}': SSE fallback also failed.`,
          );
          // Both failed without 401, throw the original error
          throw firstAttemptError;
        }
      }
    } else if (firstAttemptError) {
      // Error occurred but doesn't meet SSE fallback criteria (e.g., explicit type set)
      // Re-throw the original error
      throw firstAttemptError;
    } else {
      // This should never happen, but if we get here without firstAttemptError, throw initialError
      throw error;
    }
  }

  // This should never be reached, but TypeScript requires a return statement
  throw new Error('Unexpected end of connectToMcpServer function');
}

/**
 * Helper function to create the appropriate transport based on config
 * This handles the logic for httpUrl/url/type consistently
 */
async function createUrlTransport(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  transportOptions:
    | StreamableHTTPClientTransportOptions
    | SSEClientTransportOptions,
): Promise<StreamableHTTPClientTransport | SSEClientTransport> {
  // Priority 1: httpUrl (deprecated)
  if (mcpServerConfig.httpUrl) {
    if (mcpServerConfig.url) {
      debugLogger.warn(
        `MCP server '${mcpServerName}': Both 'httpUrl' and 'url' are configured. ` +
          `Using deprecated 'httpUrl'. Please migrate to 'url' with 'type: "http"'.`,
      );
    }
    if (!transportOptions.authProvider) {
      const oauthProvider = await getMcpOAuthClientProvider(mcpServerConfig);
      if (
        mcpServerConfig.oauth?.clientId &&
        mcpServerConfig.oauth?.clientSecret
      ) {
        const clientInformation: OAuthClientInformation = {
          client_id: mcpServerConfig.oauth.clientId,
          client_secret: mcpServerConfig.oauth.clientSecret,
        };
        oauthProvider.saveClientInformation(clientInformation);
      }
      transportOptions.authProvider = oauthProvider;
    }
    return new StreamableHTTPClientTransport(
      new URL(mcpServerConfig.httpUrl),
      transportOptions,
    );
  }

  // Priority 2 & 3: url with explicit type
  if (mcpServerConfig.url && mcpServerConfig.type) {
    if (mcpServerConfig.type === 'http') {
      return new StreamableHTTPClientTransport(
        new URL(mcpServerConfig.url),
        transportOptions,
      );
    } else if (mcpServerConfig.type === 'sse') {
      return new SSEClientTransport(
        new URL(mcpServerConfig.url),
        transportOptions,
      );
    }
  }

  // Priority 4: url without type (default to HTTP)
  if (mcpServerConfig.url) {
    return new StreamableHTTPClientTransport(
      new URL(mcpServerConfig.url),
      transportOptions,
    );
  }

  throw new Error(`No URL configured for MCP server '${mcpServerName}'`);
}

/** Visible for Testing */
export async function createTransport(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  debugMode: boolean,
  sanitizationConfig: EnvironmentSanitizationConfig,
): Promise<Transport> {
  const noUrl = !mcpServerConfig.url && !mcpServerConfig.httpUrl;
  if (noUrl) {
    if (
      mcpServerConfig.authProviderType === AuthProviderType.GOOGLE_CREDENTIALS
    ) {
      throw new Error(
        `URL must be provided in the config for Google Credentials provider`,
      );
    }
    if (
      mcpServerConfig.authProviderType ===
      AuthProviderType.SERVICE_ACCOUNT_IMPERSONATION
    ) {
      throw new Error(
        `No URL configured for ServiceAccountImpersonation MCP Server`,
      );
    }
  }
  if (mcpServerConfig.httpUrl || mcpServerConfig.url) {
    const authProvider = createAuthProvider(mcpServerConfig);
    const headers: Record<string, string> =
      (await authProvider?.getRequestHeaders?.()) ?? {};

    const transportOptions:
      | StreamableHTTPClientTransportOptions
      | SSEClientTransportOptions = {
      requestInit: createTransportRequestInit(mcpServerConfig, headers),
      authProvider,
    };

    return createUrlTransport(mcpServerName, mcpServerConfig, transportOptions);
  }

  if (mcpServerConfig.command) {
    const transport = new StdioClientTransport({
      command: mcpServerConfig.command,
      args: mcpServerConfig.args || [],
      env: {
        ...sanitizeEnvironment(process.env, sanitizationConfig),
        ...(mcpServerConfig.env || {}),
      } as Record<string, string>,
      cwd: mcpServerConfig.cwd,
      stderr: 'pipe',
    });
    if (debugMode) {
      transport.stderr!.on('data', (data) => {
        const stderrStr = data.toString().trim();
        debugLogger.debug(
          `[DEBUG] [MCP STDERR (${mcpServerName})]: `,
          stderrStr,
        );
      });
    }
    return transport;
  }

  throw new Error(
    `Invalid configuration: missing httpUrl (for Streamable HTTP), url (for SSE), and command (for stdio).`,
  );
}

interface NamedTool {
  name?: string;
}

/**
 * Cached state value for OAuth flow
 */
let cachedState: string | undefined;

/**
 * Cached OAuth provider instance per server
 */
let cachedOAuthProvider: MCPOAuthClientProvider | undefined;

/**
 * Generate PKCE parameters for OAuth flow.
 *
 * @returns PKCE state parameters
 */
export function generateStateParam(): string {
  if (cachedState) {
    return cachedState;
  }
  // Generate state for CSRF protection
  cachedState = crypto.randomBytes(16).toString('base64url');
  return cachedState;
}

export async function getMcpOAuthClientProvider(
  mcpServerConfig: MCPServerConfig,
): Promise<MCPOAuthClientProvider> {
  // Return cached provider if it exists
  if (cachedOAuthProvider) {
    return cachedOAuthProvider;
  }

  const state = generateStateParam();

  const clientMetadata: OAuthClientMetadata = {
    client_name: 'Simple OAuth MCP Client',
    redirect_uris: [CALLBACK_URL],
    grant_types: ['authorization_code', 'refresh_token'],
    response_types: ['code'],
    token_endpoint_auth_method: 'client_secret_post',
    scope: mcpServerConfig.oauth?.scopes?.join(' ') || 'openid profile email',
  };

  cachedOAuthProvider = new MCPOAuthClientProvider(
    CALLBACK_URL,
    clientMetadata,
    state,
    (authUrl: URL) => {
      console.log(`📌 OAuth redirect handler called - opening browser`);
      void openBrowser(authUrl.toString());
    },
  );

  return cachedOAuthProvider;
}

async function waitForOAuthCallback(expectedState: string): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const requestHandler = (req: IncomingMessage, res: ServerResponse) => {
      // Ignore favicon requests
      if (req.url === '/favicon.ico') {
        res.writeHead(404);
        res.end();
        return;
      }

      console.log(`📥 Received callback: ${req.url}`);
      const parsedUrl = new URL(req.url || '', 'http://localhost');
      const code = parsedUrl.searchParams.get('code');
      const error = parsedUrl.searchParams.get('error');
      const state = parsedUrl.searchParams.get('state');

      // --- 1. Authorization Error (Provider-side Failure) ---
      if (error) {
        console.log(`❌ Authorization error: ${error}`);
        res.writeHead(400, { 'Content-Type': 'text/html' });
        res.end(`
                    <html>
                      <body>
                        <h1>Authorization Failed</h1>
                        <p>Error: ${error}</p>
                      </body>
                    </html>
                `);
        server.close(); // 🛑 Close server on authorization error
        reject(new Error(`OAuth authorization failed: ${error}`));
        return;
      }

      // --- 2. Missing Parameters ---
      if (!code || !state) {
        console.log(
          `❌ Missing required parameters (code=${!!code}, state=${!!state})`,
        );
        res.writeHead(400);
        res.end('Missing authorization code or state parameter');
        server.close(); // 🛑 Close server on missing parameters
        reject(new Error('Missing required OAuth parameters'));
        return;
      }

      // --- 3. State Validation (CSRF Protection) ---
      if (state !== expectedState) {
        console.log(
          `⚠️ State mismatch! Expected: ${expectedState}, Received: ${state}`,
        );
        res.writeHead(400, { 'Content-Type': 'text/html' });
        res.end(`
                    <html>
                      <body>
                        <h1>Authorization Failed (Security Error)</h1>
                        <p>Invalid state parameter. Possible Cross-Site Request Forgery (CSRF) attempt.</p>
                      </body>
                    </html>
                `);
        server.close(); // 🛑 Close server on state mismatch
        reject(new Error('State mismatch - possible CSRF attack'));
        return;
      }

      // --- 4. Success Path ---
      console.log(
        `✅ Authorization code received: ${code.substring(0, 10)}...`,
      );

      // Send success response to the browser
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(`
                <html>
                  <body>
                    <h1>Authorization Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                    <script>setTimeout(() => window.close(), 2000);</script>
                  </body>
                </html>
            `);

      // Resolve the promise
      resolve(code);

      // Close the server on success (after a slight delay to ensure the response is sent)
      setTimeout(() => server.close(), 3000);
    };

    const server = createServer(requestHandler);

    server
      .listen(CALLBACK_PORT, () => {
        console.log(
          `OAuth callback server started on http://localhost:${CALLBACK_PORT}`,
        );
      })
      .on('error', (err: NodeJS.ErrnoException) => {
        // Handle server creation/listening errors (e.g., port in use)
        // The promise is rejected, and the application must handle cleanup if needed.
        reject(new Error(`Server listening failed: ${err.message}`));
      });
  });
}

/**
 * Opens the authorization URL in the user's default browser
 */
async function openBrowser(url: string): Promise<void> {
  console.log(`🌐 Opening browser for authorization: ${url}`);

  const command = `open "${url}"`;

  exec(command, (error) => {
    if (error) {
      console.error(`Failed to open browser: ${error.message}`);
      console.log(`Please manually open: ${url}`);
    }
  });
}

/** Visible for testing */
export function isEnabled(
  funcDecl: NamedTool,
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
): boolean {
  if (!funcDecl.name) {
    debugLogger.warn(
      `Discovered a function declaration without a name from MCP server '${mcpServerName}'. Skipping.`,
    );
    return false;
  }
  const { includeTools, excludeTools } = mcpServerConfig;

  // excludeTools takes precedence over includeTools
  if (excludeTools && excludeTools.includes(funcDecl.name)) {
    return false;
  }

  return (
    !includeTools ||
    includeTools.some(
      (tool) => tool === funcDecl.name || tool.startsWith(`${funcDecl.name}(`),
    )
  );
}
