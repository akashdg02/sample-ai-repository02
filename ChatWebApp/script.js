$(document).ready(function(){
    initializeApp();
})

function initializeApp() {
    //1. Event Listeners Setup
    setupEventListeners();
    //2. Enable Text Area Auto-Resize
    //3. Load Initial Chat History - tempororily loaded fro Browser's Local Storage
    //4. Focus Input on the TextArea
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
            e.preventDefault();
            sendMessage();
        }
    });

}

function toggleSidebar(){
    $('#sidebar').toggleClass('active');
    $('#sidebarOverlay').toggleClass('active');
}

function sendMessage(){
    console.log('You Triggered the Send Message function')
}