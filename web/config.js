/**
 * Where online (friend) games are hosted, and the key that server expects.
 *   SERVER_URL ''      this same website: right when the Python server serves the site (it probes /api/health).
 *   SERVER_URL null    no game server: only games against the agent and pass and play (static builds use this).
 *   SERVER_URL 'https://your-server.example.com'   a game server hosted elsewhere.
 * API_KEY is sent as X-API-Key when the server sets SANTORINI_API_KEYS. Anyone can read it in the page,
 * so it identifies this site rather than proving anything; the server's real protections are its origin
 * allowlist, rate limits and per-seat tokens.
 * scripts/build.mjs writes both values into dist/ from SANTORINI_SERVER_URL and SANTORINI_API_KEY.
 * The Python server answers /static/config.js itself, so this file is only used when serving web/ some other way.
 */
export const SERVER_URL = '';
export const API_KEY = '';
