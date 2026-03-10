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

      const oauthConfig = {
        ...(server.oauth ?? {}),
        enabled: true,
      };

      context.ui.addItem({
        type: 'info',
        text: `✅ Successfully authenticated with MCP server '${serverName}'!`,
      });

      // Trigger tool re-discovery to pick up authenticated server
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
        await mcpClientManager.maybeDiscoverMcpServer(
          serverName,
          updatedServerConfig,
        );
        await config.refreshMcpContext();
      }
      // Update the client with the new tools
      const geminiClient = config.getGeminiClient();
      if (geminiClient?.isInitialized()) {
        await geminiClient.setTools();
      }

      // Reload the slash commands to reflect the changes.
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

const callCommand: SlashCommand = {
  name: 'call',
  description:
    'Call an MCP server tool (e.g. /mcp call <server> with <param> <value>)',
  kind: CommandKind.BUILT_IN,
  autoExecute: true,
  action: async (
    context: CommandContext,
    args: string,
  ): Promise<MessageActionReturn> => {
    const { config } = context.services;
    if (!config) {
      return {
        type: 'message',
        messageType: 'error',
        content: 'Config not loaded.',
      };
    }

    config.setUserInteractedWithMcp();

    const trimmed = args.trim();
    if (!trimmed) {
      return {
        type: 'message',
        messageType: 'error',
        content:
          'Usage: /mcp call <server-name> with <param> <value> (example: /mcp call echo with message hi)',
      };
    }

    const withMatch = trimmed.match(
      /^([^\s]+)\s+with\s+([^\s]+)\s+([\s\S]+)$/i,
    );
    const serverName = withMatch ? withMatch[1] : trimmed.split(/\s+/)[0];
    const params: Record<string, unknown> =
      withMatch && withMatch[2] ? { [withMatch[2]]: withMatch[3] } : {};

    const mcpClientManager = config.getMcpClientManager();
    const mcpServers = mcpClientManager?.getMcpServers() ?? {};
    const server = mcpServers[serverName];
    if (!server) {
      return {
        type: 'message',
        messageType: 'error',
        content: `MCP server '${serverName}' not found.`,
      };
    }

    const sleep = (ms: number) =>
      new Promise((resolve) => setTimeout(resolve, ms));

    const ensureAuthenticatedInCall = async (): Promise<
      MessageActionReturn | undefined
    > => {
      const manager = config.getMcpClientManager();
      if (!manager) {
        return {
          type: 'message',
          messageType: 'error',
          content: 'Could not retrieve mcp client manager.',
        };
      }

      context.ui.addItem({
        type: 'info',
        text: `Starting OAuth authentication for MCP server '${serverName}'...`,
      });

      const oauthConfig = {
        ...(server.oauth ?? {}),
        enabled: true,
      };

      // Explicitly run OAuth acquisition first. Some servers (like echo) only
      // report auth failures at tool-call time and won't trigger SDK auth on connect.
      const mcpServerUrl = server.httpUrl || server.url;
      try {
        const { MCPOAuthProvider } = await import('@google/gemini-cli-core');
        const authProvider = new MCPOAuthProvider(new MCPOAuthTokenStorage());
        await authProvider.authenticate(serverName, oauthConfig, mcpServerUrl);
      } catch (error) {
        return {
          type: 'message',
          messageType: 'error',
          content: `Failed to authenticate with MCP server '${serverName}': ${getErrorMessage(error)}`,
        };
      }

      const tokenStorage = new MCPOAuthTokenStorage();
      const creds = await tokenStorage.getCredentials(serverName);
      if (!creds) {
        return {
          type: 'message',
          messageType: 'error',
          content: `OAuth finished but no credentials were stored for '${serverName}'.`,
        };
      }

      const updatedServerConfig = {
        ...server,
        oauth: oauthConfig,
      };

      try {
        await manager.maybeDiscoverMcpServer(serverName, updatedServerConfig);
        await config.refreshMcpContext();

        const geminiClient = config.getGeminiClient();
        if (geminiClient?.isInitialized()) {
          await geminiClient.setTools();
        }

        for (let index = 0; index < 12; index++) {
          if (getMCPServerStatus(serverName) === MCPServerStatus.CONNECTED) {
            return;
          }
          await sleep(250);
        }

        return {
          type: 'message',
          messageType: 'error',
          content: `MCP server '${serverName}' is not connected after authentication.`,
        };
      } catch (error) {
        return {
          type: 'message',
          messageType: 'error',
          content: `Failed to authenticate with MCP server '${serverName}': ${getErrorMessage(error)}`,
        };
      }
    };

    // Proactively start OAuth if server requires/configures OAuth but is not authenticated yet.
    if (server.oauth?.enabled || mcpServerRequiresOAuth.has(serverName)) {
      const tokenStorage = new MCPOAuthTokenStorage();
      const creds = await tokenStorage.getCredentials(serverName);
      const isExpired = !!(
        creds?.token?.expiresAt && creds.token.expiresAt < Date.now()
      );
      const isAuthenticated = !!creds && !isExpired;

      if (!isAuthenticated) {
        const authResult = await ensureAuthenticatedInCall();
        if (authResult) {
          return authResult;
        }
      }
    }

    const resolveServerTool = (): DiscoveredMCPTool | undefined => {
      const toolRegistry = config.getToolRegistry();
      const allTools = toolRegistry.getAllTools();
      return allTools.find(
        (tool): tool is DiscoveredMCPTool =>
          tool instanceof DiscoveredMCPTool && tool.serverName === serverName,
      );
    };

    let tool = resolveServerTool();
    if (!tool) {
      return {
        type: 'message',
        messageType: 'error',
        content: `No tools found for MCP server '${serverName}'.`,
      };
    }
    let result = await tool.validateBuildAndExecute(
      params,
      new AbortController().signal,
    );

    const isAuthRequiredError = (message: string) =>
      /(unauthoriz|not authenticated|oauth|authentication credential|login cookie)/i.test(
        message,
      );

    if (result.error && isAuthRequiredError(result.error.message)) {
      context.ui.addItem({
        type: 'info',
        text: `Authentication required for '${serverName}'. Starting OAuth flow...`,
      });

      const authResult = await ensureAuthenticatedInCall();
      if (authResult) {
        return authResult;
      }

      tool = resolveServerTool();
      if (!tool) {
        return {
          type: 'message',
          messageType: 'error',
          content: `No tools found for MCP server '${serverName}' after authentication.`,
        };
      }

      result = await tool.validateBuildAndExecute(
        params,
        new AbortController().signal,
      );
    }

    if (result.error) {
      return {
        type: 'message',
        messageType: 'error',
        content: `MCP call failed (${tool.name}): ${result.error.message}`,
      };
    }

    const output =
      typeof result.returnDisplay === 'string'
        ? result.returnDisplay
        : JSON.stringify(result.returnDisplay, null, 2);

    return {
      type: 'message',
      messageType: 'info',
      content: output,
    };
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
    callCommand,
    authCommand,
    reloadCommand,
    enableCommand,
    disableCommand,
  ],
  action: async (context: CommandContext) => listAction(context),
};
