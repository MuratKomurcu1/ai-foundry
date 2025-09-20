import express from 'express';
import cors from 'cors';
import nodemailer from 'nodemailer';
import Imap from 'imap';
import { simpleParser } from 'mailparser';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.GMAIL_USER,
        pass: process.env.GMAIL_APP_PASSWORD
    }
});

const createImap = () => new Imap({
    user: process.env.GMAIL_USER,
    password: process.env.GMAIL_APP_PASSWORD,
    host: 'imap.gmail.com',
    port: 993,
    tls: true,
    tlsOptions: {
        rejectUnauthorized: false
    }
});

app.post('/send', async (req, res) => {
    try {
        const { to, subject, message } = req.body;
        
        const info = await transporter.sendMail({
            from: process.env.GMAIL_USER,
            to,
            subject,
            text: message,
            html: `<p>${message.replace(/\n/g, '<br>')}</p>`
        });
        
        res.json({
            success: true,
            messageId: info.messageId,
            to,
            subject
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

app.get('/inbox', (req, res) => {
    const limit = parseInt(req.query.limit) || 5;
    const imap = createImap();
    
    imap.once('ready', () => {
        imap.openBox('INBOX', true, (err, box) => {
            if (err) {
                res.status(500).json({ success: false, error: err.message });
                return;
            }

            if (box.messages.total === 0) {
                res.json({ success: true, emails: [] });
                imap.end();
                return;
            }

            const emails = [];
            let processed = 0;
            const totalToFetch = Math.min(limit, box.messages.total);
            
            const fetch = imap.seq.fetch(`${Math.max(1, box.messages.total - limit + 1)}:*`, {
                bodies: 'HEADER.FIELDS (FROM TO SUBJECT DATE)',
                struct: true
            });

            fetch.on('message', (msg, seqno) => {
                let email = { seqno };
                
                msg.on('body', (stream) => {
                    let buffer = '';
                    stream.on('data', (chunk) => {
                        buffer += chunk.toString('utf8');
                    });
                    
                    stream.once('end', () => {
                        simpleParser(buffer, (err, parsed) => {
                            if (!err) {
                                email.from = parsed.from?.text || 'Unknown';
                                email.subject = parsed.subject || 'No Subject';
                                email.date = parsed.date ? parsed.date.toISOString() : 'Unknown';
                            }
                            
                            processed++;
                            if (processed === totalToFetch) {
                                imap.end();
                                res.json({ success: true, emails: emails.reverse() });
                            }
                        });
                    });
                });

                msg.once('end', () => {
                    emails.push(email);
                });
            });

            fetch.once('error', (err) => {
                imap.end();
                res.status(500).json({ success: false, error: err.message });
            });
        });
    });

    imap.once('error', (err) => {
        res.status(500).json({ success: false, error: err.message });
    });

    imap.connect();
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});