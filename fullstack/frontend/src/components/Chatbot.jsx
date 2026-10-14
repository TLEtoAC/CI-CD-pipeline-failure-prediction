// src/components/Chatbot.jsx — Floating AI chatbot widget
import { useState, useRef, useEffect } from "react";
import { chatAPI } from "../api";
import ReactMarkdown from "react-markdown";
import { MessageCircle, X, Send, Bot, User, Minimize2 } from "lucide-react";

const SUGGESTIONS = [
  "What does Risk Level mean?",
  "How is confidence calculated?",
  "What are the top features?",
  "How to use this dashboard?",
];

function Message({ msg }) {
  const isBot = msg.role === "assistant";
  return (
    <div className={`chat-msg ${isBot ? "bot" : "user"}`}>
      <div className="chat-avatar">{isBot ? <Bot size={14} /> : <User size={14} />}</div>
      <div className="chat-bubble">
        {isBot
          ? <ReactMarkdown components={{ p: ({children}) => <p style={{margin:"4px 0"}}>{children}</p>, strong: ({children}) => <strong style={{color:"#a5b4fc"}}>{children}</strong> }}>{msg.content}</ReactMarkdown>
          : <span>{msg.content}</span>
        }
        {msg.timestamp && <div className="chat-time">{new Date(msg.timestamp).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}</div>}
      </div>
    </div>
  );
}

export default function Chatbot() {
  const [open, setOpen]         = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, role: "assistant", content: "👋 Hi! I'm your **CI/CD Prediction Assistant**.\n\nAsk me anything about predictions, model performance, features, or how to use the dashboard!", timestamp: new Date().toISOString() }
  ]);
  const [input, setInput]   = useState("");
  const [loading, setLoading] = useState(false);
  const [unread, setUnread]   = useState(0);
  const bottomRef             = useRef(null);
  const inputRef              = useRef(null);

  useEffect(() => {
    if (open) { setUnread(0); bottomRef.current?.scrollIntoView({ behavior: "smooth" }); inputRef.current?.focus(); }
  }, [open, messages]);

  const send = async (text) => {
    const content = (text || input).trim();
    if (!content || loading) return;
    setInput("");
    const userMsg = { id: Date.now(), role: "user", content, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);
    try {
      const history = messages.slice(-6).map(m => ({ role: m.role, content: m.content }));
      const { data } = await chatAPI.send(content, history);
      const botMsg = { id: Date.now() + 1, role: "assistant", content: data.reply, timestamp: data.timestamp };
      setMessages(prev => [...prev, botMsg]);
      if (!open) setUnread(n => n + 1);
    } catch {
      setMessages(prev => [...prev, { id: Date.now()+1, role: "assistant", content: "Sorry, I couldn't connect to the server. Please try again.", timestamp: new Date().toISOString() }]);
    } finally { setLoading(false); }
  };

  const handleKey = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };

  return (
    <>
      {/* Floating button */}
      <button id="chatbot-toggle" className="chat-fab" onClick={() => setOpen(p => !p)} title="Open Chatbot">
        {open ? <X size={22} /> : <MessageCircle size={22} />}
        {!open && unread > 0 && <span className="chat-badge">{unread}</span>}
      </button>

      {/* Chat window */}
      {open && (
        <div className="chat-window glass fade-in">
          {/* Header */}
          <div className="chat-header">
            <div className="chat-header-info">
              <div className="chat-bot-avatar"><Bot size={16} /></div>
              <div>
                <div className="chat-bot-name">Pipeline Assistant</div>
                <div className="chat-status"><span className="status-dot" />Online</div>
              </div>
            </div>
            <button className="chat-close" onClick={() => setOpen(false)}><X size={16} /></button>
          </div>

          {/* Messages */}
          <div className="chat-messages">
            {messages.map(m => <Message key={m.id} msg={m} />)}
            {loading && (
              <div className="chat-msg bot">
                <div className="chat-avatar"><Bot size={14} /></div>
                <div className="chat-bubble typing"><span /><span /><span /></div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Suggestions */}
          {messages.length <= 2 && (
            <div className="chat-suggestions">
              {SUGGESTIONS.map(s => (
                <button key={s} className="suggestion-btn" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className="chat-input-row">
            <input
              ref={inputRef} id="chat-input" className="chat-input" placeholder="Ask anything…"
              value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKey}
            />
            <button id="chat-send" className="chat-send" onClick={() => send()} disabled={!input.trim() || loading}>
              <Send size={16} />
            </button>
          </div>
        </div>
      )}

      <style>{`
        .chat-fab { position: fixed; bottom: 28px; right: 28px; width: 56px; height: 56px; border-radius: 50%; background: linear-gradient(135deg,#6366f1,#4f46e5); border: none; color: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 20px rgba(99,102,241,0.5); z-index: 1000; transition: all 0.2s; animation: pulseGlow 3s ease-in-out infinite; }
        .chat-fab:hover { transform: scale(1.1); }
        .chat-badge { position: absolute; top: -4px; right: -4px; width: 20px; height: 20px; background: #ef4444; border-radius: 50%; font-size: 11px; font-weight: 700; display: flex; align-items: center; justify-content: center; color: #fff; border: 2px solid var(--bg-primary); }
        .chat-window { position: fixed; bottom: 96px; right: 28px; width: 360px; height: 520px; border-radius: var(--radius); display: flex; flex-direction: column; z-index: 1000; overflow: hidden; border: 1px solid var(--border-bright); box-shadow: 0 8px 40px rgba(0,0,0,0.5); }
        @media (max-width: 480px) { .chat-window { width: calc(100vw - 32px); right: 16px; height: 70vh; } }
        .chat-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px; background: rgba(99,102,241,0.15); border-bottom: 1px solid var(--border); }
        .chat-header-info { display: flex; align-items: center; gap: 10px; }
        .chat-bot-avatar { width: 32px; height: 32px; background: linear-gradient(135deg,#6366f1,#4f46e5); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; }
        .chat-bot-name { font-size: 13px; font-weight: 700; }
        .chat-status { display: flex; align-items: center; gap: 4px; font-size: 11px; color: #10b981; }
        .status-dot { width: 6px; height: 6px; background: #10b981; border-radius: 50%; animation: pulse 2s ease-in-out infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        .chat-close { background: none; border: none; color: var(--text-secondary); cursor: pointer; padding: 4px; border-radius: 6px; display: flex; }
        .chat-close:hover { color: var(--text-primary); background: rgba(255,255,255,0.07); }
        .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; background: rgba(10,14,26,0.8); }
        .chat-msg { display: flex; gap: 8px; align-items: flex-start; }
        .chat-msg.user { flex-direction: row-reverse; }
        .chat-avatar { width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 12px; }
        .chat-msg.bot .chat-avatar { background: rgba(99,102,241,0.2); color: #a5b4fc; }
        .chat-msg.user .chat-avatar { background: rgba(6,182,212,0.2); color: #06b6d4; }
        .chat-bubble { max-width: 85%; padding: 10px 14px; border-radius: 14px; font-size: 13px; line-height: 1.55; }
        .chat-msg.bot .chat-bubble { background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.2); border-top-left-radius: 4px; color: var(--text-primary); }
        .chat-msg.user .chat-bubble { background: linear-gradient(135deg,#6366f1,#4f46e5); border-top-right-radius: 4px; color: #fff; }
        .chat-time { font-size: 10px; color: var(--text-muted); margin-top: 5px; text-align: right; }
        .typing { display: flex; gap: 4px; align-items: center; padding: 12px 16px !important; }
        .typing span { width: 7px; height: 7px; background: #6366f1; border-radius: 50%; animation: bounce 1.2s ease-in-out infinite; }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%,80%,100%{transform:scale(0.8);opacity:0.5} 40%{transform:scale(1.2);opacity:1} }
        .chat-suggestions { padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 6px; background: rgba(0,0,0,0.2); border-top: 1px solid var(--border); }
        .suggestion-btn { background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25); color: #a5b4fc; font-size: 11px; font-weight: 600; cursor: pointer; border-radius: 20px; padding: 4px 10px; font-family: inherit; transition: all 0.2s; }
        .suggestion-btn:hover { background: rgba(99,102,241,0.2); }
        .chat-input-row { display: flex; gap: 8px; padding: 12px; background: rgba(0,0,0,0.3); border-top: 1px solid var(--border); }
        .chat-input { flex: 1; background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-radius: 22px; padding: 9px 14px; color: var(--text-primary); font-family: inherit; font-size: 13px; outline: none; transition: border-color 0.2s; }
        .chat-input:focus { border-color: var(--accent); }
        .chat-send { width: 36px; height: 36px; background: linear-gradient(135deg,#6366f1,#4f46e5); border: none; border-radius: 50%; color: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; flex-shrink: 0; }
        .chat-send:disabled { opacity: 0.5; cursor: not-allowed; }
        .chat-send:not(:disabled):hover { transform: scale(1.1); }
      `}</style>
    </>
  );
}
