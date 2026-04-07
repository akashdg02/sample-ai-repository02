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
    //4. Focus Input on the TextArea
    $("#messageInput").focus();
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
    
    generateMockResponse(messageText);
    //Show Typing Indicator
    showTypingIndicator();

    setTimeout(function(){
        const aiResponse = generateMockResponse(messageText);
        hideTypingIndicator();
        addMessage(aiResponse,'ai');
        //Save to Chat History
        saveChatToHistory();
    },1500 * Math.random());

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
    scrollToBottom();
}

function formatMessageContent(text){
    //Escape HTMl
    let formatted = $('<div>').text(text).html();

    // convert markdown-style code blocks
    formatted = formatted.replace(/```(\w+)?\n(\s\S]*?)```/g,function(match,lang,code){
        return `<pre><code>${code.trim()}</code></pre>`
    });

    //convert inline code
    formatted = formatted.replace(/`([^`]+)`/g,'<code>$1</code>');
    
    // Convert URLs to Links
    formatted = formatted.replace(
        /(https?:\/\/[^\s]+)/g,
        '<a href="$1" target="_blank">$1</a>'
    );

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
    updateChatHistoryUI();
}

function updateChatHistoryUI(){
    const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    const $chatHistory  = $("#chatHistory");
    $chatHistory.empty();

    if(history.length === 0){
        $chatHistory.html('<div style="padding:20px;text-align:center;color:var(--text-light);font-size:14px;">No chat history </div>');
        return;
    }

    history.forEach(chat => {
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
    const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    const chat = history.find(c => c.id === chatId);

    if(!chat) return;

    //Clear Current Messages
    $('.message').remove();
    $("#welcomeScreen").hide();

    //Load Message from Hisotry
    chat.messages.forEach(msg=>{
        addMessage(msg.content,msg.sender);
    });

    //Update the Title
    $("#chatTitle").text(chat.title);

    //Close Sidebar on Mobile
    if($(window).width() < 768){
        toggleSidebar();
    }
}

function getTimeAgo(date){
    const seconds = Math.floor((new Date() - date)/1000);
    if(seconds < 60) return 'Just now';
    if(seconds <3600) return Math.floor(seconds/60) + ' m ago';
    if(seconds < 86400) return Math.floor(seconds/3600) + ' h ago';
    if(seconds < 604800) return Math.floor(seconds/86400) + ' d ago';
    return date.toLocaleDateString();
}