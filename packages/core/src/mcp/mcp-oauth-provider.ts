/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as http from 'node:http';
import type { OAuthClientProvider } from '@modelcontextprotocol/sdk/client/auth.js';
import type {
  OAuthClientInformation,
  OAuthClientMetadata,
  OAuthTokens,
} from '@modelcontextprotocol/sdk/shared/auth.js';
import { URL } from 'node:url';
import type * as net from 'node:net';
import { debugLogger } from '../utils/debugLogger.js';
import * as crypto from 'node:crypto';

export const OAUTH_DISPLAY_MESSAGE_EVENT = 'oauth-display-message' as const;

const REDIRECT_PATH = '/oauth/callback';
const HTTP_OK = 200;

/**
 * OAuth authorization response.
 */
export interface OAuthAuthorizationResponse {
  code: string;
  state: string;
}

/**
 * PKCE (Proof Key for Code Exchange) parameters.
 */
interface PKCEParams {
  codeVerifier: string;
  codeChallenge: string;
  state: string;
}

export class MCPOAuthClientProvider implements OAuthClientProvider {
  private _clientInformation?: OAuthClientInformation;
  private _tokens?: OAuthTokens;
  private _codeVerifier?: string;

  constructor(
    private readonly _redirectUrl: string | URL,
    private readonly _clientMetadata: OAuthClientMetadata,
    onRedirect?: (url: URL) => void,
  ) {
    this._onRedirect =
      onRedirect ||
      ((url) => {
        console.log(`Redirect to: ${url.toString()}`);
      });
  }

  private _onRedirect: (url: URL) => void;

  get redirectUrl(): string | URL {
    return this._redirectUrl;
  }

  get clientMetadata(): OAuthClientMetadata {
    return this._clientMetadata;
  }

  clientInformation(): OAuthClientInformation | undefined {
    return this._clientInformation;
  }

  saveClientInformation(clientInformation: OAuthClientInformation): void {
    this._clientInformation = clientInformation;
  }

  tokens(): OAuthTokens | undefined {
    debugLogger.log(`Aashvi1111 get tokens`);
    return this._tokens;
  }

  saveTokens(tokens: OAuthTokens): void {
    debugLogger.log(`Aashvi1111 Saving tokens: ${JSON.stringify(tokens)}`);
    this._tokens = tokens;
  }

  /**
   * Generate PKCE parameters for OAuth flow.
   *
   * @returns PKCE parameters including code verifier, challenge, and state
   */
  generatePKCEParams(): PKCEParams {
    // Generate code verifier (43-128 characters)
    const codeVerifier = crypto.randomBytes(32).toString('base64url');

    // Generate code challenge using SHA256
    const codeChallenge = crypto
      .createHash('sha256')
      .update(codeVerifier)
      .digest('base64url');

    // Generate state for CSRF protection
    const state = crypto.randomBytes(16).toString('base64url');

    return { codeVerifier, codeChallenge, state };
  }

  async redirectToAuthorization(authorizationUrl: URL): Promise<void> {
    this._onRedirect(authorizationUrl);
  }

  saveCodeVerifier(codeVerifier: string): void {
    this._codeVerifier = codeVerifier;
  }

  codeVerifier(): string {
    if (!this._codeVerifier) {
      throw new Error('No code verifier saved');
    }
    return this._codeVerifier;
  }

  /**
   * Start a local HTTP server to handle OAuth callback.
   * The server will listen on the specified port (or port 0 for OS assignment).
   *
   * @param expectedState The state parameter to validate
   * @returns Object containing the port (available immediately) and a promise for the auth response
   */
  startCallbackServer(expectedState: string): {
    port: Promise<number>;
    response: Promise<OAuthAuthorizationResponse>;
  } {
    let portResolve: (port: number) => void;
    let portReject: (error: Error) => void;
    const portPromise = new Promise<number>((resolve, reject) => {
      portResolve = resolve;
      portReject = reject;
    });

    const responsePromise = new Promise<OAuthAuthorizationResponse>(
      (resolve, reject) => {
        let serverPort: number;

        const server = http.createServer(
          async (req: http.IncomingMessage, res: http.ServerResponse) => {
            try {
              const url = new URL(req.url!, `http://localhost:${serverPort}`);

              if (url.pathname !== REDIRECT_PATH) {
                res.writeHead(404);
                res.end('Not found');
                return;
              }

              const code = url.searchParams.get('code');
              const state = url.searchParams.get('state');
              const error = url.searchParams.get('error');

              if (error) {
                res.writeHead(HTTP_OK, { 'Content-Type': 'text/html' });
                res.end(`
              <html>
                <body>
                  <h1>Authentication Failed</h1>
                  <p>Error: ${(error as string).replace(/</g, '&lt;').replace(/>/g, '&gt;')}</p>
                  <p>${((url.searchParams.get('error_description') || '') as string).replace(/</g, '&lt;').replace(/>/g, '&gt;')}</p>
                  <p>You can close this window.</p>
                </body>
              </html>
            `);
                server.close();
                reject(new Error(`OAuth error: ${error}`));
                return;
              }

              if (!code || !state) {
                res.writeHead(400);
                res.end('Missing code or state parameter');
                return;
              }

              if (state !== expectedState) {
                res.writeHead(400);
                res.end('Invalid state parameter');
                server.close();
                reject(new Error('State mismatch - possible CSRF attack'));
                return;
              }

              // Send success response to browser
              res.writeHead(HTTP_OK, { 'Content-Type': 'text/html' });
              res.end(`
            <html>
              <body>
                <h1>Authentication Successful!</h1>
                <p>You can close this window and return to Gemini CLI.</p>
                <script>window.close();</script>
              </body>
            </html>
          `);

              server.close();
              resolve({ code, state });
            } catch (error) {
              server.close();
              reject(error);
            }
          },
        );

        server.on('error', (error) => {
          portReject(error);
          reject(error);
        });

        // Determine which port to use (env var or OS-assigned)
        const portStr = process.env['OAUTH_CALLBACK_PORT'];
        let listenPort = 0; // Default to OS-assigned port
        if (portStr) {
          const envPort = parseInt(portStr, 10);
          if (isNaN(envPort) || envPort <= 0 || envPort > 65535) {
            const error = new Error(
              `Invalid value for OAUTH_CALLBACK_PORT: "${portStr}"`,
            );
            portReject(error);
            reject(error);
            return;
          }
          listenPort = envPort;
        }

        server.listen(listenPort, () => {
          const address = server.address() as net.AddressInfo;
          serverPort = address.port;
          debugLogger.log(
            `OAuth callback server listening on port ${serverPort}`,
          );
          portResolve(serverPort); // Resolve port promise immediately
        });

        // Timeout after 5 minutes
        setTimeout(
          () => {
            server.close();
            reject(new Error('OAuth callback timeout'));
          },
          5 * 60 * 1000,
        );
      },
    );

    return { port: portPromise, response: responsePromise };
  }
}
