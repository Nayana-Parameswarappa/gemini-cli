/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  MCPOAuthClientProvider,
  OAUTH_DISPLAY_MESSAGE_EVENT,
  type OAuthAuthorizationResponse,
} from './mcp-oauth-provider.js';
import type {
  OAuthClientInformation,
  OAuthClientMetadata,
  OAuthTokens,
} from '@modelcontextprotocol/sdk/shared/auth.js';

describe('MCPOAuthClientProvider', () => {
  const mockRedirectUrl = 'http://localhost:8090/callback';
  const mockClientMetadata: OAuthClientMetadata = {
    client_name: 'Test Client',
    redirect_uris: [mockRedirectUrl],
    grant_types: ['authorization_code', 'refresh_token'],
    response_types: ['code'],
    token_endpoint_auth_method: 'client_secret_post',
    scope: 'test-scope',
  };
  const mockState = 'test-state-123';

  describe('constructor', () => {
    it('should create provider with required parameters', () => {
      const provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
      );

      expect(provider).toBeInstanceOf(MCPOAuthClientProvider);
      expect(provider.redirectUrl).toBe(mockRedirectUrl);
      expect(provider.clientMetadata).toEqual(mockClientMetadata);
    });

    it('should use custom onRedirect handler when provided', async () => {
      const onRedirectMock = vi.fn();
      const provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
        onRedirectMock,
      );

      const testUrl = new URL('http://auth.example.com/authorize');
      await provider.redirectToAuthorization(testUrl);

      expect(onRedirectMock).toHaveBeenCalledWith(testUrl);
    });

    it('should use default console.log when onRedirect not provided', async () => {
      const consoleLogSpy = vi
        .spyOn(console, 'log')
        .mockImplementation(() => {});
      const provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
      );

      const testUrl = new URL('http://auth.example.com/authorize');
      await provider.redirectToAuthorization(testUrl);

      expect(consoleLogSpy).toHaveBeenCalledWith(
        `Redirect to: ${testUrl.toString()}`,
      );
      consoleLogSpy.mockRestore();
    });

    it('should accept URL object for redirectUrl', () => {
      const urlObject = new URL(mockRedirectUrl);
      const provider = new MCPOAuthClientProvider(
        urlObject,
        mockClientMetadata,
        mockState,
      );

      expect(provider.redirectUrl).toBe(urlObject);
    });

    it('should work without state parameter', () => {
      const provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
      );

      expect(provider).toBeInstanceOf(MCPOAuthClientProvider);
    });
  });

  describe('clientInformation', () => {
    let provider: MCPOAuthClientProvider;

    beforeEach(() => {
      provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
      );
    });

    it('should return undefined when no client information is saved', () => {
      expect(provider.clientInformation()).toBeUndefined();
    });

    it('should save and retrieve client information', () => {
      const clientInfo: OAuthClientInformation = {
        client_id: 'test-client-id',
        client_secret: 'test-client-secret',
      };

      provider.saveClientInformation(clientInfo);
      expect(provider.clientInformation()).toEqual(clientInfo);
    });

    it('should overwrite previous client information', () => {
      const clientInfo1: OAuthClientInformation = {
        client_id: 'client-1',
        client_secret: 'secret-1',
      };
      const clientInfo2: OAuthClientInformation = {
        client_id: 'client-2',
        client_secret: 'secret-2',
      };

      provider.saveClientInformation(clientInfo1);
      provider.saveClientInformation(clientInfo2);
      expect(provider.clientInformation()).toEqual(clientInfo2);
    });
  });

  describe('tokens', () => {
    let provider: MCPOAuthClientProvider;

    beforeEach(() => {
      provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
      );
    });

    it('should return undefined when no tokens are saved', () => {
      expect(provider.tokens()).toBeUndefined();
    });

    it('should save and retrieve tokens', () => {
      const tokens: OAuthTokens = {
        access_token: 'test-access-token',
        token_type: 'Bearer',
        expires_in: 3600,
        refresh_token: 'test-refresh-token',
      };

      provider.saveTokens(tokens);
      expect(provider.tokens()).toEqual(tokens);
    });

    it('should overwrite previous tokens', () => {
      const tokens1: OAuthTokens = {
        access_token: 'token-1',
        token_type: 'Bearer',
        expires_in: 3600,
      };
      const tokens2: OAuthTokens = {
        access_token: 'token-2',
        token_type: 'Bearer',
        expires_in: 7200,
        refresh_token: 'refresh-token-2',
      };

      provider.saveTokens(tokens1);
      provider.saveTokens(tokens2);
      expect(provider.tokens()).toEqual(tokens2);
    });
  });

  describe('codeVerifier', () => {
    let provider: MCPOAuthClientProvider;

    beforeEach(() => {
      provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
      );
    });

    it('should throw error when no code verifier is saved', () => {
      expect(() => provider.codeVerifier()).toThrow('No code verifier saved');
    });

    it('should save and retrieve code verifier', () => {
      const verifier = 'test-code-verifier-abc123';
      provider.saveCodeVerifier(verifier);
      expect(provider.codeVerifier()).toBe(verifier);
    });

    it('should overwrite previous code verifier', () => {
      provider.saveCodeVerifier('verifier-1');
      provider.saveCodeVerifier('verifier-2');
      expect(provider.codeVerifier()).toBe('verifier-2');
    });
  });

  describe('state', () => {
    it('should return state when provided in constructor', () => {
      const provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
      );

      expect(provider.state()).toBe(mockState);
    });

    it('should throw error when no state is provided', () => {
      const provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
      );

      expect(() => provider.state()).toThrow('No code state saved');
    });
  });

  describe('callback server', () => {
    let provider: MCPOAuthClientProvider;

    beforeEach(() => {
      provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
      );
    });

    it('should return undefined when no callback server is saved', () => {
      expect(provider.getSavedCallbackServer()).toBeUndefined();
    });

    it('should save and retrieve callback server', async () => {
      const mockResponse: OAuthAuthorizationResponse = {
        code: 'auth-code-123',
        state: mockState,
      };
      const mockServer = {
        port: Promise.resolve(8090),
        waitForResponse: vi.fn().mockResolvedValue(mockResponse),
        close: vi.fn().mockResolvedValue(undefined),
      };

      provider.saveCallbackServer(mockServer);
      const savedServer = provider.getSavedCallbackServer();

      expect(savedServer).toBe(mockServer);
      expect(await savedServer?.port).toBe(8090);
      expect(await savedServer?.waitForResponse()).toEqual(mockResponse);
    });

    it('should overwrite previous callback server', () => {
      const mockServer1 = {
        port: Promise.resolve(8090),
        waitForResponse: vi.fn(),
        close: vi.fn(),
      };
      const mockServer2 = {
        port: Promise.resolve(8091),
        waitForResponse: vi.fn(),
        close: vi.fn(),
      };

      provider.saveCallbackServer(mockServer1);
      provider.saveCallbackServer(mockServer2);
      expect(provider.getSavedCallbackServer()).toBe(mockServer2);
    });
  });

  describe('redirectToAuthorization', () => {
    it('should call onRedirect with authorization URL', async () => {
      const onRedirectMock = vi.fn();
      const provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
        onRedirectMock,
      );

      const authUrl = new URL(
        'http://auth.example.com/authorize?client_id=123',
      );
      await provider.redirectToAuthorization(authUrl);

      expect(onRedirectMock).toHaveBeenCalledExactlyOnceWith(authUrl);
    });
  });

  describe('OAUTH_DISPLAY_MESSAGE_EVENT constant', () => {
    it('should have correct value', () => {
      expect(OAUTH_DISPLAY_MESSAGE_EVENT).toBe('oauth-display-message');
    });
  });

  describe('integration scenario', () => {
    it('should support full OAuth flow', async () => {
      const onRedirectMock = vi.fn();
      const provider = new MCPOAuthClientProvider(
        mockRedirectUrl,
        mockClientMetadata,
        mockState,
        onRedirectMock,
      );

      // Step 1: Save client information
      const clientInfo: OAuthClientInformation = {
        client_id: 'my-client-id',
        client_secret: 'my-client-secret',
      };
      provider.saveClientInformation(clientInfo);

      // Step 2: Save code verifier
      provider.saveCodeVerifier('my-code-verifier');

      // Step 3: Set up callback server
      const mockAuthResponse: OAuthAuthorizationResponse = {
        code: 'authorization-code',
        state: mockState,
      };
      const mockServer = {
        port: Promise.resolve(8090),
        waitForResponse: vi.fn().mockResolvedValue(mockAuthResponse),
        close: vi.fn().mockResolvedValue(undefined),
      };
      provider.saveCallbackServer(mockServer);

      // Step 4: Redirect to authorization
      const authUrl = new URL('http://auth.example.com/authorize');
      await provider.redirectToAuthorization(authUrl);

      // Step 5: Save tokens after exchange
      const tokens: OAuthTokens = {
        access_token: 'final-access-token',
        token_type: 'Bearer',
        expires_in: 3600,
        refresh_token: 'final-refresh-token',
      };
      provider.saveTokens(tokens);

      // Verify all data is stored correctly
      expect(provider.clientInformation()).toEqual(clientInfo);
      expect(provider.codeVerifier()).toBe('my-code-verifier');
      expect(provider.state()).toBe(mockState);
      expect(provider.tokens()).toEqual(tokens);
      expect(onRedirectMock).toHaveBeenCalledWith(authUrl);
      expect(provider.getSavedCallbackServer()).toBe(mockServer);
    });
  });
});
