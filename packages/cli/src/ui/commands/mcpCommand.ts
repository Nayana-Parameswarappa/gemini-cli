/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  SlashCommand,
  SlashCommandActionReturn,
  CommandContext,
} from './types.js';
import { CommandKind } from './types.js';
import type { MessageActionReturn } from '@google/gemini-cli-core';
import {
  DiscoveredMCPTool,
  getMCPDiscoveryState,
  getMCPServerStatus,
  MCPDiscoveryState,
  MCPServerStatus,
  getErrorMessage,
  MCPOAuthTokenStorage,
  MCPOAuthProvider,
  mcpServerRequiresOAuth,
  CoreEvent,
  coreEvents,
} from '@google/gemini-cli-core';

import { MessageType, type HistoryItemMcpStatus } from '../types.js';
import {
  McpServerEnablementManager,
  normalizeServerId,
  canLoadServer,
} from '../../config/mcp/mcpServerEnablement.js';
import { loadSettings } from '../../config/settings.js';

const MCP_SDK_OAUTH_ENABLED =
  process.env['GEMINI_CLI_ENABLE_MCP_SDK_OAUTH'] !== '0';

const MCP_AUTH_WAIT_TIMEOUT_MS = 2 * 60 * 1000;
const MCP_AUTH_WAIT_POLL_INTERVAL_MS = 250;
const MCP_AUTH_DISCOVERY_TIMEOUT_MS = 90 * 1000;
const MCP_AUTH_CONNECTED_CHECK_TIMEOUT_MS = 8 * 1000;

async function waitForMcpServerConnected(
  serverName: string,
  mcpClientManager:
    | ReturnType<
        NonNullable<CommandContext['services']['config']>['getMcpClientManager']
      >
    | undefined,
  timeoutMs: number = MCP_AUTH_WAIT_TIMEOUT_MS,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (getMCPServerStatus(serverName) === MCPServerStatus.CONNECTED) {
      return true;
    }

    await new Promise((resolve) =>
      setTimeout(resolve, MCP_AUTH_WAIT_POLL_INTERVAL_MS),
    );
  }

  const managerError =
    typeof mcpClientManager?.getLastError === 'function'
      ? mcpClientManager.getLastError(serverName)
      : undefined;

  if (managerError) {
    throw new Error(managerError);
  }

  return false;
}

async function runMcpDiscoveryWithTimeout<TConfig>(
  serverName: string,
  maybeDiscoverMcpServer: (name: string, config: TConfig) => Promise<void>,
  updatedServerConfig: TConfig,
): Promise<{ completed: boolean; discoveryPromise: Promise<void> }> {
  const discoveryPromise = maybeDiscoverMcpServer(
    serverName,
    updatedServerConfig,
  );

  const completed = await Promise.race([
    discoveryPromise.then(() => true),
    new Promise<boolean>((resolve) => {
      setTimeout(() => {
        resolve(false);
      }, MCP_AUTH_DISCOVERY_TIMEOUT_MS);
    }),
  ]);

  return { completed, discoveryPromise };
}

const authCommand: SlashCommand = {
  name: 'auth',
  description: 'Authenticate with an OAuth-enabled MCP server',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: async (
    context: CommandContext,
    args: string,
  ): Promise<MessageActionReturn> => {
    const serverName = args.trim();
    const { config } = context.services;

    if (!config) {
      return {
        type: 'message',
        messageType: 'error',
        content: 'Config not loaded.',
      };
    }

    config.setUserInteractedWithMcp();

    const mcpServers = config.getMcpClientManager()?.getMcpServers() ?? {};

    if (!serverName) {
      // List servers that support OAuth from two sources:
      // 1. Servers with oauth.enabled in config
      // 2. Servers detected as requiring OAuth (returned 401)
      const configuredOAuthServers = Object.entries(mcpServers)
        .filter(([_, server]) => server.oauth?.enabled)
        .map(([name, _]) => name);

      const detectedOAuthServers = Array.from(
        mcpServerRequiresOAuth.keys(),
      ).filter((name) => mcpServers[name]); // Only include configured servers

      // Combine and deduplicate
      const allOAuthServers = [
        ...new Set([...configuredOAuthServers, ...detectedOAuthServers]),
      ];

      if (allOAuthServers.length === 0) {
        return {
          type: 'message',
          messageType: 'info',
          content: 'No MCP servers configured with OAuth authentication.',
        };
      }

      return {
        type: 'message',
        messageType: 'info',
        content: `MCP servers with OAuth authentication:\n${allOAuthServers.map((s) => `  - ${s}`).join('\n')}\n\nUse /mcp auth <server-name> to authenticate.`,
      };
    }

    const server = mcpServers[serverName];
    if (!server) {
      return {
        type: 'message',
        messageType: 'error',
        content: `MCP server '${serverName}' not found.`,
      };
    }

    // Always attempt OAuth authentication, even if not explicitly configured
    // The authentication process will discover OAuth requirements automatically

    const displayListener = (message: string) => {
      context.ui.addItem({ type: 'info', text: message });
    };

    coreEvents.on(CoreEvent.OauthDisplayMessage, displayListener);
    try {
      context.ui.addItem({
        type: 'info',
        text: `Starting OAuth authentication for MCP server '${serverName}'...`,
      });

      if (!MCP_SDK_OAUTH_ENABLED) {
        let oauthConfig = server.oauth;
        if (!oauthConfig) {
          oauthConfig = { enabled: false };
        }

        const mcpServerUrl = server.httpUrl || server.url;
        const authProvider = new MCPOAuthProvider(new MCPOAuthTokenStorage());
        await authProvider.authenticate(serverName, oauthConfig, mcpServerUrl);

        context.ui.addItem({
          type: 'info',
          text: `✅ Successfully authenticated with MCP server '${serverName}'!`,
        });

        const mcpClientManager = config.getMcpClientManager();
        if (mcpClientManager) {
          context.ui.addItem({
            type: 'info',
            text: `Restarting MCP server '${serverName}'...`,
          });
          await mcpClientManager.restartServer(serverName);
        }

        const geminiClient = config.getGeminiClient();
        if (geminiClient?.isInitialized()) {
          await geminiClient.setTools();
        }

        context.ui.reloadCommands();

        return {
          type: 'message',
          messageType: 'info',
          content: `Successfully authenticated and refreshed tools for '${serverName}'.`,
        };
      }

      const oauthConfig = {
        ...(server.oauth ?? {}),
        enabled: true,
      };

      context.ui.addItem({
        type: 'info',
        text: `Waiting for OAuth callback from MCP server '${serverName}'...`,
      });

      const mcpClientManager = config.getMcpClientManager();
      if (mcpClientManager) {
        context.ui.addItem({
          type: 'info',
          text: `Restarting MCP server '${serverName}'...`,
        });
        const updatedServerConfig = {
          ...server,
          oauth: oauthConfig,
        };
        const { completed, discoveryPromise } =
          await runMcpDiscoveryWithTimeout(
            serverName,
            mcpClientManager.maybeDiscoverMcpServer.bind(mcpClientManager),
            updatedServerConfig,
          );

        if (!completed) {
          context.ui.addItem({
            type: 'info',
            text: `OAuth completed for '${serverName}'. MCP reconnect is still running in the background...`,
          });

          void discoveryPromise
            .then(async () => {
              await config.refreshMcpContext();

              const geminiClient = config.getGeminiClient();
              if (geminiClient?.isInitialized()) {
                await geminiClient.setTools();
              }

              context.ui.reloadCommands();
              context.ui.addItem({
                type: 'info',
                text: `✅ MCP server '${serverName}' is ready.`,
              });
            })
            .catch((error: unknown) => {
              context.ui.addItem({
                type: 'error',
                text: `MCP reconnect failed for '${serverName}': ${getErrorMessage(error)}`,
              });
            });

          return {
            type: 'message',
            messageType: 'info',
            content: `Authentication completed for '${serverName}'. MCP tool refresh continues in the background.`,
          };
        }

        await config.refreshMcpContext();
      }

      const isConnected = await waitForMcpServerConnected(
        serverName,
        mcpClientManager,
        MCP_AUTH_CONNECTED_CHECK_TIMEOUT_MS,
      );

      if (!isConnected) {
        context.ui.addItem({
          type: 'info',
          text: `OAuth completed for '${serverName}'. Final MCP readiness checks continue in the background...`,
        });

        void (async () => {
          try {
            await waitForMcpServerConnected(serverName, mcpClientManager);
            await config.refreshMcpContext();

            const geminiClient = config.getGeminiClient();
            if (geminiClient?.isInitialized()) {
              await geminiClient.setTools();
            }

            context.ui.reloadCommands();
            context.ui.addItem({
              type: 'info',
              text: `✅ MCP server '${serverName}' is ready.`,
            });
          } catch (error: unknown) {
            context.ui.addItem({
              type: 'error',
              text: `MCP finalization failed for '${serverName}': ${getErrorMessage(error)}`,
            });
          }
        })();

        return {
          type: 'message',
          messageType: 'info',
          content: `Authentication completed for '${serverName}'. MCP tool refresh continues in the background.`,
        };
      }

      const geminiClient = config.getGeminiClient();
      if (geminiClient?.isInitialized()) {
        await geminiClient.setTools();
      }

      context.ui.reloadCommands();

      return {
        type: 'message',
        messageType: 'info',
        content: `Successfully authenticated and refreshed tools for '${serverName}'.`,
      };
    } catch (error) {
      return {
        type: 'message',
        messageType: 'error',
        content: `Failed to authenticate with MCP server '${serverName}': ${getErrorMessage(error)}`,
      };
    } finally {
      coreEvents.removeListener(CoreEvent.OauthDisplayMessage, displayListener);
    }
  },
  completion: async (context: CommandContext, partialArg: string) => {
    const { config } = context.services;
    if (!config) return [];

    const mcpServers = config.getMcpClientManager()?.getMcpServers() || {};
    return Object.keys(mcpServers).filter((name) =>
      name.startsWith(partialArg),
    );
  },
};

const listAction = async (
  context: CommandContext,
  showDescriptions = false,
  showSchema = false,
): Promise<void | MessageActionReturn> => {
  const { config } = context.services;
  if (!config) {
    return {
      type: 'message',
      messageType: 'error',
      content: 'Config not loaded.',
    };
  }

  config.setUserInteractedWithMcp();

  const toolRegistry = config.getToolRegistry();
  if (!toolRegistry) {
    return {
      type: 'message',
      messageType: 'error',
      content: 'Could not retrieve tool registry.',
    };
  }

  const mcpServers = config.getMcpClientManager()?.getMcpServers() || {};
  const serverNames = Object.keys(mcpServers);
  const blockedMcpServers =
    config.getMcpClientManager()?.getBlockedMcpServers() || [];

  const connectingServers = serverNames.filter(
    (name) => getMCPServerStatus(name) === MCPServerStatus.CONNECTING,
  );
  const discoveryState = getMCPDiscoveryState();
  const discoveryInProgress =
    discoveryState === MCPDiscoveryState.IN_PROGRESS ||
    connectingServers.length > 0;

  const allTools = toolRegistry.getAllTools();
  const mcpTools = allTools.filter((tool) => tool instanceof DiscoveredMCPTool);

  const promptRegistry = config.getPromptRegistry();
  const mcpPrompts = promptRegistry
    .getAllPrompts()
    .filter(
      (prompt) =>
        'serverName' in prompt && serverNames.includes(prompt.serverName),
    );

  const resourceRegistry = config.getResourceRegistry();
  const mcpResources = resourceRegistry
    .getAllResources()
    .filter((entry) => serverNames.includes(entry.serverName));

  const authStatus: HistoryItemMcpStatus['authStatus'] = {};
  const tokenStorage = new MCPOAuthTokenStorage();
  for (const serverName of serverNames) {
    const server = mcpServers[serverName];
    // Check auth status for servers with oauth.enabled OR detected as requiring OAuth
    if (server.oauth?.enabled || mcpServerRequiresOAuth.has(serverName)) {
      const creds = await tokenStorage.getCredentials(serverName);
      if (creds) {
        if (creds.token.expiresAt && creds.token.expiresAt < Date.now()) {
          authStatus[serverName] = 'expired';
        } else {
          authStatus[serverName] = 'authenticated';
        }
      } else {
        authStatus[serverName] = 'unauthenticated';
      }
    } else {
      authStatus[serverName] = 'not-configured';
    }
  }

  // Get enablement state for all servers
  const enablementManager = McpServerEnablementManager.getInstance();
  const enablementState: HistoryItemMcpStatus['enablementState'] = {};
  for (const serverName of serverNames) {
    enablementState[serverName] =
      await enablementManager.getDisplayState(serverName);
  }
  const errors: Record<string, string> = {};
  for (const serverName of serverNames) {
    const error = config.getMcpClientManager()?.getLastError(serverName);
    if (error) {
      errors[serverName] = error;
    }
  }

  const mcpStatusItem: HistoryItemMcpStatus = {
    type: MessageType.MCP_STATUS,
    servers: mcpServers,
    tools: mcpTools.map((tool) => ({
      serverName: tool.serverName,
      name: tool.name,
      description: tool.description,
      schema: tool.schema,
    })),
    prompts: mcpPrompts.map((prompt) => ({
      serverName: prompt.serverName,
      name: prompt.name,
      description: prompt.description,
    })),
    resources: mcpResources.map((resource) => ({
      serverName: resource.serverName,
      name: resource.name,
      uri: resource.uri,
      mimeType: resource.mimeType,
      description: resource.description,
    })),
    authStatus,
    enablementState,
    errors,
    blockedServers: blockedMcpServers.map((s) => ({
      name: s.name,
      extensionName: s.extensionName,
    })),
    discoveryInProgress,
    connectingServers,
    showDescriptions: Boolean(showDescriptions),
    showSchema: Boolean(showSchema),
  };

  context.ui.addItem(mcpStatusItem);
};
const listCommand: SlashCommand = {
  name: 'list',
  altNames: ['ls', 'nodesc', 'nodescription'],
  description: 'List configured MCP servers and tools',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: (context) => listAction(context),
};

const descCommand: SlashCommand = {
  name: 'desc',
  altNames: ['description'],
  description: 'List configured MCP servers and tools with descriptions',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: (context) => listAction(context, true),
};

const schemaCommand: SlashCommand = {
  name: 'schema',
  description:
    'List configured MCP servers and tools with descriptions and schemas',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: (context) => listAction(context, true, true),
};

const reloadCommand: SlashCommand = {
  name: 'reload',
  altNames: ['refresh'],
  description: 'Reloads MCP servers',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: async (
    context: CommandContext,
  ): Promise<void | SlashCommandActionReturn> => {
    const { config } = context.services;
    if (!config) {
      return {
        type: 'message',
        messageType: 'error',
        content: 'Config not loaded.',
      };
    }

    const mcpClientManager = config.getMcpClientManager();
    if (!mcpClientManager) {
      return {
        type: 'message',
        messageType: 'error',
        content: 'Could not retrieve mcp client manager.',
      };
    }

    context.ui.addItem({
      type: 'info',
      text: 'Reloading MCP servers...',
    });

    await mcpClientManager.restart();

    // Update the client with the new tools
    const geminiClient = config.getGeminiClient();
    if (geminiClient?.isInitialized()) {
      await geminiClient.setTools();
    }

    // Reload the slash commands to reflect the changes.
    context.ui.reloadCommands();

    return listCommand.action!(context, '');
  },
};

async function handleEnableDisable(
  context: CommandContext,
  args: string,
  enable: boolean,
): Promise<MessageActionReturn> {
  const { config } = context.services;
  if (!config) {
    return {
      type: 'message',
      messageType: 'error',
      content: 'Config not loaded.',
    };
  }

  config.setUserInteractedWithMcp();

  const parts = args.trim().split(/\s+/);
  const isSession = parts.includes('--session');
  const serverName = parts.filter((p) => p !== '--session')[0];
  const action = enable ? 'enable' : 'disable';

  if (!serverName) {
    return {
      type: 'message',
      messageType: 'error',
      content: `Server name required. Usage: /mcp ${action} <server-name> [--session]`,
    };
  }

  const name = normalizeServerId(serverName);

  // Validate server exists
  const servers = config.getMcpClientManager()?.getMcpServers() || {};
  const normalizedServerNames = Object.keys(servers).map(normalizeServerId);
  if (!normalizedServerNames.includes(name)) {
    return {
      type: 'message',
      messageType: 'error',
      content: `Server '${serverName}' not found. Use /mcp list to see available servers.`,
    };
  }

  const manager = McpServerEnablementManager.getInstance();

  if (enable) {
    const settings = loadSettings();
    const result = await canLoadServer(name, {
      adminMcpEnabled: settings.merged.admin?.mcp?.enabled ?? true,
      allowedList: settings.merged.mcp?.allowed,
      excludedList: settings.merged.mcp?.excluded,
    });
    if (
      !result.allowed &&
      (result.blockType === 'allowlist' || result.blockType === 'excludelist')
    ) {
      return {
        type: 'message',
        messageType: 'error',
        content: result.reason ?? 'Blocked by settings.',
      };
    }
    if (isSession) {
      manager.clearSessionDisable(name);
    } else {
      await manager.enable(name);
    }
    if (result.blockType === 'admin') {
      context.ui.addItem(
        {
          type: 'warning',
          text: 'MCP disabled by admin. Will load when enabled.',
        },
        Date.now(),
      );
    }
  } else {
    if (isSession) {
      manager.disableForSession(name);
    } else {
      await manager.disable(name);
    }
  }

  const msg = `MCP server '${name}' ${enable ? 'enabled' : 'disabled'}${isSession ? ' for this session' : ''}.`;

  const mcpClientManager = config.getMcpClientManager();
  if (mcpClientManager) {
    context.ui.addItem(
      { type: 'info', text: 'Reloading MCP servers...' },
      Date.now(),
    );
    await mcpClientManager.restart();
  }
  if (config.getGeminiClient()?.isInitialized())
    await config.getGeminiClient().setTools();
  context.ui.reloadCommands();

  return { type: 'message', messageType: 'info', content: msg };
}

async function getEnablementCompletion(
  context: CommandContext,
  partialArg: string,
  showEnabled: boolean,
): Promise<string[]> {
  const { config } = context.services;
  if (!config) return [];
  const servers = Object.keys(
    config.getMcpClientManager()?.getMcpServers() || {},
  );
  const manager = McpServerEnablementManager.getInstance();
  const results: string[] = [];
  for (const n of servers) {
    const state = await manager.getDisplayState(n);
    if (state.enabled === showEnabled && n.startsWith(partialArg)) {
      results.push(n);
    }
  }
  return results;
}

const enableCommand: SlashCommand = {
  name: 'enable',
  description: 'Enable a disabled MCP server',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: (ctx, args) => handleEnableDisable(ctx, args, true),
  completion: (ctx, arg) => getEnablementCompletion(ctx, arg, false),
};

const disableCommand: SlashCommand = {
  name: 'disable',
  description: 'Disable an MCP server',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: (ctx, args) => handleEnableDisable(ctx, args, false),
  completion: (ctx, arg) => getEnablementCompletion(ctx, arg, true),
};

export const mcpCommand: SlashCommand = {
  name: 'mcp',
  description: 'Manage configured Model Context Protocol (MCP) servers',
  kind: CommandKind.BUILT_IN,
  autoExecute: false,
  subCommands: [
    listCommand,
    descCommand,
    schemaCommand,
    authCommand,
    reloadCommand,
    enableCommand,
    disableCommand,
  ],
  action: async (context: CommandContext) => listAction(context),
};
