# Privacy Policy — Clairwave MCP server

Effective 2026-09-03. Applies to the remote MCP server at `https://www.clairwave.com/mcp`,
operated by Clairwave (https://www.clairwave.com).

## What the server is
An open, no-login endpoint that lets AI assistants query Clairwave's ocean-acoustics
platform: bathymetry, sound-speed climatology, seabed parameters, acoustic propagation
simulations, vessel source-level estimates, and live AIS vessel positions. No user
account is created and no credentials are requested from the user.

## Data we collect
For each tool call the server records:
- the tool name and the arguments passed (for example coordinates, frequency, an MMSI
  or vessel name);
- the caller's IP address and HTTP user agent (as forwarded by our CDN);
- timing, success/failure and error text;
- the identifier of any simulation run created on the platform.

We do not receive or store the surrounding conversation, prompts, or any content other
than the tool arguments the assistant sends. Simulation outputs are stored on the
platform under a random run id and are publicly retrievable by anyone who has that id.

## How we use it
Operational monitoring, abuse prevention, capacity planning and aggregate usage
statistics (calls per tool, per client type, per day). We do not build profiles of
individual users and do not use the data for advertising.

## Sharing
We do not sell or share the collected data with third parties. Requests are served
through Cloudflare (CDN and DDoS protection), which processes IP addresses under its
own privacy policy. Vessel data originates from the AISHub network and public sources.

## Retention
Tool-call logs are retained for up to 12 months and then deleted. Simulation run files
may be retained indefinitely as reproducible scientific artifacts; they contain no
personal data.

## Your choices
The server is anonymous; there is nothing to log in to or delete. If you believe a log
entry identifies you and want it removed, contact us with the approximate time and tool.

## Contact
contact@clairwave.com
