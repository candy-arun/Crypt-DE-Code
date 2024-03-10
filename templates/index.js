let menuicn = document.querySelector(".menuicn"); 
let nav = document.querySelector(".navcontainer"); 

menuicn.addEventListener("click", () => { 
	nav.classList.toggle("navclose"); 
})
// JavaScript code (index.js)
$(document).ready(function() {
    $('.option2').click(function() { // Changed 'nav-option' to '.nav-option'
        $.ajax({
            url: '/livetransaction.py',
            type: 'GET',
            success: function(data) {
                // Handle the response data and update the HTML page
                console.log(data); // Output the received data to console for now
            },
            error: function(xhr, status, error) {
                console.error('Error:', error);
            }
        });
    });
});

