const API_BASE_URL = 'http://127.0.0.1:5000/api'; // Placeholder for backend API URL
let currentConversationId = null; // To track the current conversation context

$(document).ready(function(){
    initializeApp();
})

function initializeApp() {
    //1. Event Listeners Setup
    setupEventListeners();
    //2. Enable Text Area Auto-Resize
    autoResizeTextArea();
    //3. Load Initial Chat History - tempororily loaded fro Browser's Local Storage
    loadChatHistory();
    //4. Check API Conneciton
    checkApiConnection();
    //5. Focus Input o(n the TextArea
    $("#messageInput").focus();
}

function checkApiConnection(){
    const url = `${API_BASE_URL}/health`;
    console.log(url)
    $.ajax({
        url:url,
        method:'GET',
        timeout:10000,
        success:function(response){
            console.log("API Connection Successful",response);
            hideConnectionError();
        },
        error:function(xhr){
            console.error("API Connection Failed",xhr);
            showConnectionError();
        }
    })
}

function showConnectionError(){
    const errorBanner = $(`
        <div id="connectionError" style="position:fixed;top:0;left:0;right:0;background:#f44336;color:white;padding:12px;text-align:center;z-index:9999;font-weight:500;">
            Connection to API failed. Please check your internet connection and try again
            <button onclick="checkAPIConnection()" style="margin-left:20px;padding:6px 12px;background:white;color:#f44336;border:none;border-radius:4px;cursor:pointer;font-weight:500;">Retry</button>
        </div>
    `);
    $("#connectionError").remove();
    $("body").append(errorBanner);
}

function hideConnectionError(){
    $("#connectionError").remove();
}

function setupEventListeners() {
    //Sidebar Toggle
    $('#sidebarToggle').on('click',toggleSidebar);
    $('#sidebarOverlay').on('click',toggleSidebar);

    //Enable or Disable Send Button based on TextArea Input
    $('#messageInput').on('input', function(){
        const hasText = $(this).val().trim().length > 0;
        $('#sendBtn').prop('disabled', !hasText);
    });

    //Show coming soon feature for File Attachement
    $('#attachBtn').on('click', function(){
        alert('File attachment feature coming soon!');
    });

    //Enable Click on Send Button Press
    $('#sendBtn').on('click',sendMessage)

    $("#messageInput").on('keypress', function(e){
        if(e.key === 'Enter' && !e.shiftKey){
            if($('#sendBtn').prop('disabled')===false){
                e.preventDefault();
                console.log()
                sendMessage();
            }
        }
    });

    // Start New Chat
    $("#newChatBtn").on('click',startNewChat);

    //Clear Existing Chat
    $("#clearChatBtn").on('click',clearCurrentChat);

    //Suggestions Cards
    $(".suggestion-card").on('click',function(){
        const prompt = $(this).data('prompt');
        $("#messageInput").val(prompt);
        $("#sendBtn").prop('disabled',false);
        sendMessage();
    });
}

function toggleSidebar(){
    $('#sidebar').toggleClass('active');
    $('#sidebarOverlay').toggleClass('active');
}

function sendMessage(){
    const messageText = $("#messageInput").val().trim();
    // Non Empty Check
    if(!messageText) return;
    console.log(messageText)
    // Hide the welcome Scren
    $("#welcomeScreen").hide();
    //Add user message to the list and UI
    addMessage(messageText,'user');
    //Clear the current Input Box
    $("#messageInput").val('').css('height','auto')
    $("#sendBtn").prop('disabled',true);
    
    //generateMockResponse(messageText);
    //Show Typing Indicator
    showTypingIndicator();

    $.ajax({
        url:`${API_BASE_URL}/chat`,
        method:'POST',
        contentType:'application/json',
        data:JSON.stringify({
            message:messageText,
            conversation_id:currentConversationId
        }),
        success:function(response){
            hideTypingIndicator();
            console.log(response);
            if(response.status){
                console.log("Received response from API",response);
                currentConversationId = response.data.conversation_id; // Update conversation ID for context
                addMessage(response.data.response,'assistant');
                updateChatTitle(respopnse.data.conversation_id);
                loadChatHistory(); // Refresh chat history to reflect any title changes
            }else{
                addMessage("Sorry, something went wrong. Please try again later.",'error');
            }
        },
        error:function(xhr,status,error){
            hideTypingIndicator();
            let errorMessage = "Failed to get response from API. Please check your connection and try again.";
            if(xhr.responseJSON && xhr.responseJSON.error){
                errorMessage = xhr.responseJSON.error;
            }else if(xhr.status === 0){
                errorMessage = "Unable to connect to the API.check if flask is running";
                showConnectionError();
            }else if(xhr.status === 500){
                errorMessage = "Server error occurred. Please try again later.";
            }

            addMessage('Error : ' + errorMessage,'error');
            console.error("API Error",xhr,status,error);
        }
    })

    //Scroll to the bottom of the chat
    scrollToBottom();
}

function addMessage(text,sender){
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    const senderName = sender === 'user' ? 'You' : 'AI Assistant';
    const senderInitial = sender === 'user' ? 'Y' : 'AI';

    const messageHTML = `
    <div class="message ${sender}">
        <div class="message-header">
            <div class="message-avatar">${senderInitial}</div>
            <span class="message-sender">${senderName}</span>
            <span class="message-time">${time}</span>
        </div>
        <div class="message-content">
            ${formatMessageContent(text)}
        </div>
    </div>
    `;
    $('#typingIndicator').before(messageHTML);

    if(typeof hljs !== 'undefined'){
        $('#messagesWrapper .message:last pre code').each(function(i, block) {
            hljs.highlightElement(block);
        })
    }

    scrollToBottom();
}

function formatMessageContent(text){

    if(typeof marked !== 'undefined'){
        marked.setOptions({
            breaks:true,
            gfm:true,
            headerIds:false,
            mangle:false
        });
        return marked.parse(text);
    }

    return parseMarkdown(text);
}

function parseMarkdown(text){
    //Escape HTMl
    let formatted = $('<div>').text(text).html();

    // convert markdown-style code blocks
    formatted = formatted.replace(/```(\w+)?\n(\s\S]*?)```/g,function(match,lang,code){
        const language = lang || 'plaintext';
        return `<pre><code class="language-${language}">${code.trim()}</code></pre>`
    });

    //convert inline code
    formatted = formatted.replace(/`([^`]+)`/g,'<code>$1</code>');

    //Bold text 
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
    formatted = formatted.replace(/__([^_]+)__/g,'<strong>$1</strong>');

    //Italic Text
    formatted = formatted.replace(/\*([^*]+)\*/g,'<em>$1</em>');
    formatted = formatted.replace(/_([^_]+)_/g,'<em>$1</em>');

    //Strikethough
    formatted = formatted.replace(/~~([^~]+)~~/g,'<del>$1</del>');

    //Headings ###,##, #
    formatted = formatted.replace(/^### (.+)$/gm,'<h3>$1</h3>');
    formatted = formatted.replace(/^## (.+)$/gm,'<h2>$1</h2>');
    formatted = formatted.replace(/^# (.+)$/gm,'<h1>$1</h1>');

    //Unordered Lists 
    formatted = formatted.replace(/^\* (.+)&/gm,'___UL_ITEM___$1___UL_ITEM___');
    formatted = formatted.replace(/^\- (.+)&/gm,'___UL_ITEM___$1___UL_ITEM___');
    formatted = formatted.replace(/^\• (.+)&/gm,'___UL_ITEM___$1___UL_ITEM___');

    // Wrap list items in <ul> tags
    formatted = formatted.replace(/___UL_ITEM___([\s\S]+?)___UL_ITEM___/g,function(match,items){
        const listItems = items.trim().split('___UL_ITEM___').map(item => `<li>${item.trim()}</li>`).join('');
        return `<ul>${listItems}</ul>`;
    });

    //Ordered Lists
    formatted = formatted.replace(/^\d+\. (.+)$/gm,'___OL_ITEM___$1___OL_ITEM___');
    formatted = formatted.replace(/___OL_ITEM___([\s\S]+?)___OL_ITEM___/g,function(match,items){
        const listItems = items.trim().split('___OL_ITEM___').map(item => `<li>${item.trim()}</li>`).join('');
        return `<ol>${listItems}</ol>`;
    });

    //Links
    formatted = formatted.replace(/\[([^\]]+)\]\((https?:\/\/[^\s]+)\)/g,'<a href="$2" target="_blank">$1</a>');

    // Auto-link URLs
    formatted = formatted.replace(
        /(?<!href="|href=')(https?:\/\/[^\s<]+)(?<!">)/g,
        '<a href="$1" target="_blank">$1</a>'
    );

    //Blockquotes
    formatted = formatted.replace(/^&gt (.+)$/gm,'<blockquote>$1</blockquote>');

    //Horizontal Rule
    formatted = formatted.replace(/^---$/gm,'<hr>');
    formatted = formatted.replace(/^\*\*\*$/gm,'<hr>');
    formatted = formatted.replace(/^\=\=\=$/gm,'<hr>');



    //convert line breaks
    formatted = formatted.replace(/\n/g,'<br>');
    return formatted;

}

function showTypingIndicator(){
    $("#typingIndicator").addClass('active');
    scrollToBottom();
}

function hideTypingIndicator(){
    $("#typingIndicator").removeClass('active');
}

function scrollToBottom(){
    const container = $("#messagesContainer");
    container.animate({
        scrollTop: container[0].scrollHeight
    }, 300);
}

function autoResizeTextArea(){
    $("#messageInput").on('input',function(){
        this.style.height= 'auto';
        this.style.height = (this.scrollHeight)+'px';
    });
}

function startNewChat(){
    if(confirm('Start a New Chat? Current Chat will be saved.')){
        // Save Current Chat
        saveChatToHistory();
        // Clear Current Chat Messages
        $('.message').remove();
        // Show t he Welcome Screen
        $("#welcomeScreen").show();
        //Update the Title
        $("#chatTitle").text('New Chat');
        // Close Sidebar on Mobile
        if($(window).width() < 768){
            toggleSidebar();
        }
    }
}

function clearCurrentChat(){
    if(confirm('Clear all messages')){
        $('.message').remove();
         // Show t he Welcome Screen
         $("#welcomeScreen").show();
         //Update the Title
         $("#chatTitle").text('New Chat');
    }
}

function generateMockResponse(userMessage){
    console.log('triggered')
     const responses = [
        "That's a great question! Let me help you with that.\n\nHere's what you need to know:\n\n1. First point about your question\n2. Second important aspect\n3. Additional considerations\n\nWould you like me to elaborate on any of these points?",
        
        "I understand what you're asking. Here's a comprehensive answer:\n\n" + userMessage + "\n\nBased on that, I can provide several insights that might be helpful for your situation.",
        
        "Excellent question! Let me break this down for you:\n\n```python\n# Example code\ndef example_function():\n    return 'This is how it works'\n```\n\nThis demonstrates the concept you're asking about.",
        
        "I'd be happy to help with that! Here are some key points:\n\n• Point one\n• Point two\n• Point three\n\nLet me know if you need more details on any of these!",
        
        "That's an interesting topic. Here's my perspective:\n\nThe main thing to understand is the relationship between these concepts. When you consider the broader context, it becomes clearer how everything fits together.\n\nDoes this make sense?"
    ];
    return responses[Math.floor(Math.random() * responses.length)];

}
function saveChatToHistory(){
    const messages = [];
    $('.message').each(function(){
        const sender = $(this).hasClass('user') ? 'user' : 'assistant';
        const content = $(this).find(".message-content").text();
        messages.push({sender,content});
    })

    if(messages.length === 0) return ;

    const firstUserMessage = messages.find(msg => msg.sender === 'user');
    const title = firstUserMessage ? firstUserMessage.content.substring(0,30) + '...' : 'New Chat';

    //Get the current Chat History from Local Storage
    let history = JSON.parse(localStorage.getItem('chatHistory') || '[]');

    //Add New Chat 
    history.unshift({
        id:Date.now(),
        title:title,
        messages:messages,
        timestamp: new Date().toISOString()
    })

    //Keep only the latest 20 Chats in History
    history = history.slice(0,20);

    // Save to Local Storage
    localStorage.setItem('chatHistory',JSON.stringify(history));

    //Update UI
    updateChatHistoryUI();
}

function loadChatHistory(){
    //updateChatHistoryUI();
    $.ajax({
        url:`${API_BASE_URL}/conversations`,
        method:'GET',
        success:function(response){
            if(response.success){
                updateChatHistoryUI(response.data.conversations);
            }
        },
        error:function(xhr){
            console.log("Failed to load chat history",xhr);
            const $chatHistory  = $("#chatHistory");
            $chatHistory.html('<div style="padding:20px;text-align:center;color:var(--text-light);font-size:14px;">Failed to load chat history</div>');

        }
    })
}

function updateChatHistoryUI(conversations){
  
    const $chatHistory  = $("#chatHistory");
    $chatHistory.empty();
    console.log(conversations);
    if(conversations.length === 0){
        $chatHistory.html('<div style="padding:20px;text-align:center;color:var(--text-light);font-size:14px;">No chat history </div>');
        return;
    }

    conversations.forEach(chat => {
        const date = new Date(chat.timestamp);
        const timeAge = getTimeAgo(date);
        const $item = $(`
            <div class="chat-history-item" data-chat-id="${chat.id}">
                <i class="fas fa-message"></i>
                <span>${chat.title}</span>
            </div>
        `);
        $item.on('click',function(){
            loadChat(chat.id);
        });
        $chatHistory.append($item);
    });
}

function loadChat(chatId){
    $.ajax({
        url:`${API_BASE_URL}/conversations/${chatId}`,
        method:'GET',
        success:function(response){
            if(response.success){
                $('.message').remove();
                $("#welcomeScreen").hide();

                currentConversationId = chatId;

                //Load Message from Hisotry
                response.data.messages.forEach(msg=>{
                    addMessage(msg.content,msg.role);
                });

                if(typeof hljs !== 'undefined'){
                    $('#messagesWrapper .message:last pre code').each(function(i, block) {
                        hljs.highlightElement(block);
                    })
                }

                //Update the Title
                $("#chatTitle").text(response.data.title||'Chat');

                //Close Sidebar on Mobile
                if($(window).width() < 768){
                    toggleSidebar();
                }
            }
        },
        error:function(xhr){
            console.log("Failed to load chat",xhr);
            alert('Failed to load chat history. Please try again later.');
        }
    });

}

function updateChatTitle(conversationId){
    $.ajax({
        url:`${API_BASE_URL}/conversations/${conversationId}`,
        method:'GET',
        success:function(response){
            if(response.success && response.data.title){
                $("#chatTitle").text(response.data.title);
            }
        },
        error:function(xhr){
                console.log("Failed to update chat title",xhr);
        }
    });
}

function getTimeAgo(date){
    const seconds = Math.floor((new Date() - date)/1000);
    if(seconds < 60) return 'Just now';
    if(seconds <3600) return Math.floor(seconds/60) + ' m ago';
    if(seconds < 86400) return Math.floor(seconds/3600) + ' h ago';
    if(seconds < 604800) return Math.floor(seconds/86400) + ' d ago';
    return date.toLocaleDateString();
}