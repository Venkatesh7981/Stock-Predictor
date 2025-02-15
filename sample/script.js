document.getElementById('contact-form').addEventListener('submit', function(e) {
    e.preventDefault();
    alert('Message sent! Thank you for contacting me.');
    this.reset(); // Reset the form fields
});