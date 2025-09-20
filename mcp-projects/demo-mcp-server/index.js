#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
    name: "ping-server",
    version: "1.0.0",
});

// Tool kaydı - eksik syntax düzeltildi
server.registerTool("pingWebsite", 
    {
        title: "Ping Website tool",
        description: "Sends a ping request to a given URL and returns the result",
        inputSchema: { url: z.string().url() }
    },
    // Handler fonksiyonu buraya gelir
    async ({ url }) => {
        const { exec } = await import("child_process");
        
        const host = url.replace(/^https?:\/\//, '').split('/')[0];
        
        return new Promise((resolve) => {
            // Windows için ping komutu (-n parametresi)
            const pingCommand = process.platform === 'win32' ? `ping -n 2 ${host}` : `ping -c 2 ${host}`;
            
            exec(pingCommand, (error, stdout, stderr) => {
                if (error) {
                    resolve({
                        content: [{
                            type: "text", // "test" değil "text" olmalı
                            text: `Ping Failed: ${stderr || error.message}`
                        }]
                    });
                } else {
                    resolve({
                        content: [{
                            type: "text", // "test" değil "text" olmalı
                            text: stdout
                        }]
                    });
                }
            });
        });
    }
);

// Server'ı başlat
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
}

main().catch(console.error);