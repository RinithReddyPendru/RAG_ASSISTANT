const API_BASE = window.location.protocol === 'file:' || (window.location.port !== '8000' && window.location.hostname !== '') ? "http://localhost:8000/api" : "/api";

document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const apiKeyInput = document.getElementById("api-key");
    const uploadStatus = document.getElementById("upload-status");
    const sendBtn = document.getElementById("send-btn");

    // Load model from local storage
    const savedKey = localStorage.getItem("ollama_model");
    if (savedKey) {
        apiKeyInput.value = savedKey;
    }

    apiKeyInput.addEventListener("change", (e) => {
        localStorage.setItem("ollama_model", e.target.value);
    });

    // Chat functionality
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        const apiKey = apiKeyInput.value.trim();
        
        if (!message) return;
        
        if (!apiKey) {
            appendMessage("assistant", "⚠️ Please enter your Ollama model name (e.g. 'llama3') in the sidebar.");
            return;
        }

        // Add user message
        appendMessage("user", message);
        chatInput.value = "";
        
        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message, api_key: apiKey })
            });
            
            removeTypingIndicator(typingId);
            
            if (response.ok) {
                const data = await response.json();
                appendMessage("assistant", data.response.trim());
            } else {
                const errorData = await response.json();
                appendMessage("assistant", `❌ Error: ${errorData.detail || 'Internal server error'}`);
            }
        } catch (error) {
            removeTypingIndicator(typingId);
            appendMessage("assistant", "❌ Network error: Could not connect to the backend server. Make sure it's running.");
            console.error(error);
        }
    });

    // File Upload handling
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.endsWith('.pdf') && !file.name.endsWith('.txt')) {
            showStatus('Error: Only PDF and TXT files are allowed.', 'error');
            return;
        }
        
        showStatus(`Uploading and processing ${file.name}...`, '');
        
        const formData = new FormData();
        formData.append("file", file);
        
        const currentModel = apiKeyInput.value.trim() || 'llama3';
        formData.append("model_name", currentModel);
        
        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: "POST",
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                showStatus(`✅ Success: ${file.name} ingested (${data.chunks} segments).`, 'success');
                appendMessage("assistant", `I have successfully read **${file.name}**. You can now ask me questions about it!`);
            } else {
                const errorData = await response.json();
                showStatus(`❌ Error: ${errorData.detail}`, 'error');
            }
        } catch (error) {
            showStatus('❌ Network error: Backend server may not be running.', 'error');
            console.error(error);
        }
    }

    function showStatus(msg, type) {
        uploadStatus.textContent = msg;
        uploadStatus.className = `status-msg ${type}`;
        uploadStatus.style.display = "block";
        if (type === 'success') {
            setTimeout(() => {
                uploadStatus.style.display = "none";
            }, 5000);
        }
    }

    function appendMessage(role, content) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}`;
        
        const avatarUrl = role === 'user' 
            ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
            : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7L12 12L22 7L12 2Z"></path></svg>';
            
        const formattedContent = role === 'assistant' && window.marked ? marked.parse(content) : `<p>${escapeHTML(content)}</p>`;
        
        msgDiv.innerHTML = `
            <div class="avatar ${role}-avatar">
                ${avatarUrl}
            </div>
            <div class="message-content">
                ${formattedContent}
            </div>
        `;
        
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.className = `message assistant`;
        msgDiv.id = id;
        
        msgDiv.innerHTML = `
            <div class="avatar assistant-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7L12 12L22 7L12 2Z"></path></svg>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }
});
