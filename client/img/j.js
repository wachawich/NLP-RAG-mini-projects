const messageBar = document.querySelector(".bar-wrapper input");
const sendBtn = document.querySelector(".bar-wrapper button");
const messageBox = document.querySelector(".message-box");

let API_URL = "http://127.0.0.1:5000/api/process_message";

const typeWriter = (text, element) => {
  let i = 0;
  const interval = setInterval(() => {
    element.innerHTML += text.charAt(i);
    i++;
    if (i > text.length) {
      clearInterval(interval);
    }
  }, 10); // Adjust typing speed here (milliseconds per character)
};
// Function to send message
const sendMessage = () => {
  if (messageBar.value.length > 0) {
    const UserTypedMessage = messageBar.value.trim();
    messageBar.value = "";

    let message = `<div class="chat message">
      <span>
        ${UserTypedMessage}
      </span>
      <img src="img/user.jpg">
    </div>`;

    let response = `<div class="chat response">
      <img src="img/chatbot.jpg">
      <span class="new">...</span>
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
          const ChatBotResponse = document.querySelector(".response .new");
          ChatBotResponse.classList.remove("new");
          typeWriter(data.response, ChatBotResponse);
        })
        .catch((error) => {
          const ChatBotResponse = document.querySelector(".response .new");
          ChatBotResponse.innerHTML = "Oops! An error occurred. Please try again";
        });
    }, 200);
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
