(function () {
  "use strict";

  // Configuration
  const CHATBOT_API_URL = "https://bbs-fitment.vercel.app"; // Replace with your actual URL

  // Create container for chatbot
  const container = document.createElement("div");
  container.id = "BBS-chatbot-container";
  document.body.appendChild(container);

  // Capture shop domain from the parent page
  const shopDomain = window.location.hostname;

  // Load the chatbot iframe
  const iframe = document.createElement("iframe");
  iframe.src =
    CHATBOT_API_URL + "/chatbot-embed?shop=" + encodeURIComponent(shopDomain);
  iframe.style.cssText = `
    position: fixed;
    bottom: 0;
    right: 0;
    width: 100px;
    height: 100px;
    border: none;
    z-index: 999999;
    background: transparent;
    pointer-events: none;
    transition: width 0.3s, height 0.3s;
  `;
  iframe.setAttribute("id", "BBS-chatbot-iframe");

  container.appendChild(iframe);

  // Listen for messages from the chatbot
  window.addEventListener("message", function (event) {
    // Security: verify origin matches your app
    if (event.origin !== CHATBOT_API_URL) {
      return;
    }

    if (event.data && event.data.type === "chatbot") {
      if (event.data.isOpen) {
        // Chat is open - expand to fit chat window (approx 450px width, 850px height)
        iframe.style.width = "450px";
        iframe.style.height = "850px";
        iframe.style.pointerEvents = "auto";
      } else {
        // Chat is closed - shrink to just the button size
        iframe.style.width = "100px";
        iframe.style.height = "100px";
        iframe.style.pointerEvents = "auto";
      }
    }
  });
})();
