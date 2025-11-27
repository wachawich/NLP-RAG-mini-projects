const messageBar = document.querySelector(".bar-wrapper input");
const sendBtn = document.querySelector(".bar-wrapper button");
const messageBox = document.querySelector(".message-box");

let API_URL = "http://127.0.0.1:5000/api/qa";


// Function to send message
const sendMessage = () => {
  if (messageBar.value.length > 0) {
    const UserTypedMessage = messageBar.value.trim();
    messageBar.value = "";

    let message = `<div class="chat message">
      <span>
        ${UserTypedMessage}
      </span>
    </div>`;

    let response = `<div class="chat response">
      <img src="img/chatbot.jpg">
      <span class="new">...
      </span>
    </div>`;

    messageBox.insertAdjacentHTML("beforeend", message);

    setTimeout(() => {
      messageBox.insertAdjacentHTML("beforeend", response);

      const requestOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: UserTypedMessage }),
      };

      fetch(API_URL, requestOptions)
        .then((res) => res.json())
        .then((data) => {
          try {
            // แปลง Markdown เป็น HTML
            const htmlContent = marked.parse(data.response);
    
            // กำหนด HTML ที่แปลงแล้วให้กับ element
            const ChatBotResponse = document.querySelector(".response .new");
            ChatBotResponse.innerHTML = htmlContent;
            ChatBotResponse.classList.remove("new");
          } catch (error) {
            console.error('Error processing markdown:', error);
            const ChatBotResponse = document.querySelector(".response .new");
            ChatBotResponse.innerHTML = "Oops! An error occurred while processing markdown.";
          }
        })
        .catch((error) => {
          const ChatBotResponse = document.querySelector(".response .new");
          ChatBotResponse.innerHTML = "Oops! An error occurred. Please try again";
        });
    }, 100);
  }
};

// Click event on button
sendBtn.addEventListener("click", sendMessage);

// Enter key press event on input field
messageBar.addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendMessage(); // Call sendMessage function
  }
});
