// Authorization header necessary when we want to authorize someone to do something.
let api = "http://localhost:5000";

let toggle_modal_button = document.querySelector("#Toggle_login");
let sign_in_btn = document.querySelector("#login");
let currentUserEmail = null;
let currentTopic = "CS";

let chatForm = null;
let chatInput = null;
let csBtn = null;
let medBtn = null;

let login_form = null;
let toglle_button = null;
let authmode = "signup";

/* ---------- AUTH FORM ---------- */

function initAuthForm() {
  // grab the elements now that DOM is loaded
  login_form = document.getElementById("login-form");
  toglle_button = document.getElementById("toggle-auth");
  sign_in_btn = document.getElementById("login");

  login_form.onsubmit = function (event) {
    event.preventDefault();
    console.log("submit login/signup");

    let emailInput = document.getElementById("email").value.trim();
    let usernameInput = document.getElementById("username").value.trim();
    let passwordInput = document.getElementById("password").value.trim();

    if (authmode == "signup") {
      doSignUp(emailInput, usernameInput, passwordInput);
    } else {
      console.log("logging in");
      doLogin(emailInput, passwordInput);
    }
  };

  toglle_button.onclick = function () {
    if (authmode == "signup") {
      setAuthMode("login");
    } else {
      setAuthMode("signup");
    }
  };

  setAuthMode("signup");
}

function setAuthMode(mode) {
  authmode = mode;

  let title = document.getElementById("auth-tittle");
  let usernameGroup = document.getElementById("username-group");
  let toggleText = document.getElementById("auth-toggle-text");
  let toggleBtn = document.getElementById("toggle-auth");
  let filler = document.getElementById("username");

  if (mode == "signup") {
    title.textContent = "Sign Up";
    sign_in_btn.textContent = "Sign Up";
    usernameGroup.style.display = "block";
    toggleText.firstChild.textContent = "Already have an account? ";
    toggleBtn.textContent = "Log In";
    filler.value = "";
  } else {
    title.textContent = "Log In";
    sign_in_btn.textContent = "Log In";
    usernameGroup.style.display = "none";
    toggleText.firstChild.textContent = "Need an account? ";
    toggleBtn.textContent = "Sign Up";
    filler.value = "filler"; // to satisfy "required" when hidden
  }
}

function doSignUp(email, username, password) {
  let data = "email=" + encodeURIComponent(email);
  data += "&username=" + encodeURIComponent(username);
  data += "&password=" + encodeURIComponent(password);

  fetch(api + "/users", {
    body: data,
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Authorization: authorizationHeader(),
    },
  })
    .then(function (res) {
      if (!res.ok) {
        return res.text().then(function (text) {
          showMessage(text || "Signup failed", "error");
          throw new Error(text || "Signup failed");
        });
      }
      return res.text();
    })
    .then(function (text) {
      console.log("signup good:", text);
      createSessionID();
      currentUserEmail = email;
      localStorage.setItem("email", email);

      showChatForUser(email);
      initChatUI();
      showMessage("Account created successfully!", "success");
      clearAuthFields();
    })
    .catch(function (err) {
      console.log("signup error:", err);
    });
}

function doLogin(email, password) {
  let data = "email=" + encodeURIComponent(email);
  data += "&password=" + encodeURIComponent(password);

  fetch(api + "/users/login", {
    body: data,
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
  })
    .then(function (response) {
      if (!response.ok) {
        showMessage("Invalid email or password", "error");
        throw new Error("login failed");
      }
      return response.text(); // or response.json() depending on backend
    })
    .then(function (text) {
      console.log("login ok:", text);
      createSessionID();
      currentUserEmail = email;
      localStorage.setItem("email", email);

      showChatForUser(email);
      initChatUI();
      showMessage("Logged in successfully!", "success");
      clearAuthFields();
    })
    .catch(function (error) {
      console.log("login error:", error);
    });
}

function clearAuthFields() {
  document.getElementById("email").value = "";
  document.getElementById("username").value = "";
  document.getElementById("password").value = "";
}

/* ---------- MESSAGES (TOASTS) ---------- */

function showMessage(message, status) {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }

  let toast = document.createElement("div");
  toast.className =
    "toast " + (status === "success" ? "toast-success" : "toast-error");
  toast.textContent = message;

  container.appendChild(toast);

  setTimeout(function () {
    toast.classList.add("show");
  }, 10);

  // hide and remove after 3 seconds
  setTimeout(function () {
    toast.classList.remove("show");
    setTimeout(function () {
      container.removeChild(toast);
      if (!container.hasChildNodes()) {
        container.parentNode.removeChild(container);
      }
    }, 200);
  }, 3000);
}

document.addEventListener(
  "submit",
  function (event) {
    console.log("A FORM TRIED TO SUBMIT:", event.target);

    if (event.target.id === "chat-form") {
      event.preventDefault();
      event.stopImmediatePropagation();
      console.log("Blocked chat-form refresh at document level");
    }
  },
  true,
);
/* ---------- CHAT UI ---------- */

function initChatUI() {
  chatForm = document.getElementById("chat-form");
  chatInput = document.getElementById("chat-input");
  csBtn = document.getElementById("topic-cs-btn");
  medBtn = document.getElementById("topic-med-btn");

  if (!chatForm || !chatInput) {
    console.log("Missing chat form or chat input");
    return;
  }

  // This blocks the form from ever doing a real browser submit
  chatForm.addEventListener("submit", function (event) {
    event.preventDefault();
    event.stopPropagation();
    console.log("Blocked normal form submit");
    return false;
  });

  let sendBtn = document.querySelector(".chat-send-btn");

  if (sendBtn) {
    sendBtn.type = "button"; // forces it not to submit the form

    sendBtn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();

      console.log("Send button clicked");
      sendChatMessage();

      return false;
    });
  }

  if (csBtn) {
    csBtn.onclick = function () {
      setTopic("CS");
    };
  }

  if (medBtn) {
    medBtn.onclick = function () {
      setTopic("MED");
    };
  }
}

function setTopic(topic) {
  currentTopic = topic;
  csBtn.classList.remove("active");
  medBtn.classList.remove("active");

  if (currentTopic == "CS") {
    csBtn.classList.add("active");
  } else {
    medBtn.classList.add("active");
  }
}

function showChatForUser(email) {
  let chatPanel = document.getElementById("chat-panel");
  let label = document.getElementById("chat-user-label");

  currentUserEmail = email;
  label.textContent = email;

  chatPanel.style.display = "flex";
  chatPanel.classList.remove("hidden");
}

function removeChatForUser() {
  let chatPanel = document.getElementById("chat-panel");
  let label = document.getElementById("chat-user-label");
  let messages = document.getElementById("chat-messages");

  chatPanel.classList.add("hidden");
  chatPanel.style.display = "none";

  currentUserEmail = null;
  if (label) {
    label.textContent = "";
  }
  if (messages) {
    messages.innerHTML = "";
  }
}

function addMessage(role, text, topic) {
  let container = document.getElementById("chat-messages");
  let row = document.createElement("div");

  // "message-row user" or "message-row assistant"
  row.className = "message-row " + role;

  let bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (role === "user" && topic) {
    let tag = document.createElement("div");
    tag.className = "message-topic-tag";
    tag.textContent = topic;
    bubble.appendChild(tag);
  }

  let body = document.createElement("div");
  body.textContent = text;
  bubble.appendChild(body);
  row.appendChild(bubble);
  container.appendChild(row);

  container.scrollTop = container.scrollHeight;
}

function sendChatMessage() {
  let text = chatInput.value.trim();
  console.log("hello");

  if (!text) {
    return;
  }
  if (!isLoggedIn()) {
    alert("You must be logged in");
    return;
  }
  addMessage("user", text, currentTopic);
  chatInput.value = "";
  addTypingIndicator();
  logInteraction(currentUserEmail, text, currentTopic);
}

/* ---------- INTERACTIONS ---------- */

function logInteraction(email, query, topic) {
  let data = "email=" + encodeURIComponent(email);
  data += "&query=" + encodeURIComponent(query);
  data += "&topic=" + encodeURIComponent(topic);

  console.log("Sending interaction data:", data);

  fetch(api + "/interactions", {
    method: "POST",
    body: data,
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Authorization: authorizationHeader(),
    },
  })
    .then(function (response) {
      console.log("Interaction response status:", response.status);
      console.log("Interaction response url:", response.url);
      console.log("Interaction redirected:", response.redirected);

      return response.text().then(function (text) {
        console.log("Raw server response:", text);

        if (!response.ok) {
          throw new Error(text || "Failed to get response from server");
        }

        try {
          return JSON.parse(text);
        } catch (err) {
          throw new Error("Server did not return JSON. It returned: " + text);
        }
      });
    })
    .then(function (data) {
      console.log("Parsed server data:", data);

      removeTypingIndicator();

      let reply = data.reply || data.message || data.answer;

      if (!reply) {
        addMessage(
          "assistant",
          "The server responded, but there was no reply field.",
        );
        console.log("Missing reply field. Full server data:", data);
        return;
      }

      addMessage("assistant", reply);
    })
    .catch(function (err) {
      console.log("logInteraction error:", err);

      removeTypingIndicator();
      addMessage("assistant", "Something went wrong: " + err.message);
    });
}

/* ---------- LOGIN / LOGOUT / SESSIONS ---------- */

toggle_modal_button.onclick = function () {
  if (isLoggedIn()) {
    SignOut();
  } else {
    openLoginModal();
  }
};

function SignOut() {
  // tell the backend to drop the session
  deleteSession();

  // clear local session + email
  localStorage.setItem("sessionID", "");
  localStorage.removeItem("email");

  // hide chat UI
  removeChatForUser();

  console.log("sessionID after signout:", localStorage.getItem("sessionID"));

  // update banner button text
  check();
}

function addTypingIndicator() {
  var container = document.getElementById("chat-messages");

  var row = document.createElement("div");
  row.className = "message-row assistant";
  row.id = "typing-indicator";

  var bubble = document.createElement("div");
  bubble.className = "message-bubble typing-bubble typing-dots";
  bubble.textContent = "Thinking";

  row.appendChild(bubble);
  container.appendChild(row);

  container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
  var indicator = document.getElementById("typing-indicator");
  if (indicator) {
    indicator.remove();
  }
}

function authorizationHeader() {
  let sessionID = localStorage.getItem("sessionID");
  if (sessionID) {
    // console.log("Found a session id in authheader");
    return "Bearer " + sessionID;
  } else {
    return null;
  }
}

function isLoggedIn() {
  let session = localStorage.getItem("sessionID");
  // console.log("sessionID stored:", session);
  return session !== null && session !== "";
}

function createSessionID() {
  fetch(api + "/sessions", {
    headers: {
      Authorization: authorizationHeader(),
    },
  }).then(function (response) {
    if (response.status == 200) {
      response.json().then(function (session) {
        localStorage.setItem("sessionID", session.id);
        closeLoginModal();
        check();
      });
    }
  });
}

function deleteSession() {
  fetch(api + "/sessions", {
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Authorization: authorizationHeader(),
    },
    method: "DELETE",
  }).then(function (response) {
    console.log("deleted session");
  });
}

/* ---------- MODAL + UI STATE ---------- */

function openLoginModal() {
  document.getElementById("loginModal").style.display = "block";
}

function closeLoginModal() {
  document.getElementById("loginModal").style.display = "none";
}

window.onclick = function (event) {
  if (event.target == document.getElementById("loginModal")) {
    closeLoginModal();
  }
};

function check() {
  if (isLoggedIn()) {
    closeLoginModal();
    toggle_modal_button.textContent = "Sign Out";
  } else {
    toggle_modal_button.textContent = "Sign In";
  }
}
/* ---------- MODE TOGGLE: CHAT vs QUIZ ---------- */

function initModeToggle() {
  var chatModeBtn = document.getElementById("chat-mode-btn");
  var quizModeBtn = document.getElementById("quiz-mode-btn");
  var chatPanel = document.getElementById("chat-panel");
  var quizView = document.getElementById("quiz-view");

  if (!chatModeBtn || !quizModeBtn || !chatPanel || !quizView) {
    return;
  }

  chatModeBtn.onclick = function () {
    // Show chat, hide quiz
    chatPanel.classList.remove("hidden");
    chatPanel.style.display = "flex";
    quizView.classList.add("hidden");
  };

  quizModeBtn.onclick = function () {
    if (!isLoggedIn()) {
      alert("You must be logged in to use quiz mode.");
      return;
    }
    // Show quiz, hide chat
    quizView.classList.remove("hidden");
    chatPanel.classList.add("hidden");
    chatPanel.style.display = "none";
  };
}

/* ---------- QUIZ MODE UI ---------- */

function initQuizUI() {
  var quizView = document.getElementById("quiz-view");
  var topicSelect = document.getElementById("quiz-topic");
  var includeHistoryCheckbox = document.getElementById("include-history");
  var requestQuizBtn = document.getElementById("request-quiz-btn");
  var quizRequestInput = document.getElementById("quiz-request-text");
  var quizContainer = document.getElementById("quiz-container");
  var quizListDiv = document.getElementById("quiz-list"); // NEW

  if (
    !quizView ||
    !topicSelect ||
    !includeHistoryCheckbox ||
    !requestQuizBtn ||
    !quizRequestInput ||
    !quizContainer
  ) {
    return;
  }

  requestQuizBtn.onclick = function () {
    if (!isLoggedIn()) {
      alert("You must be logged in to request a quiz.");
      return;
    }
    if (!currentUserEmail) {
      alert("Missing email; please log in again.");
      return;
    }

    var topic = topicSelect.value;
    var includeHistory = includeHistoryCheckbox.checked ? "yes" : "no";
    var quizRequest = quizRequestInput.value.trim();
    if (!quizRequest) {
      alert("Please describe what you want the quiz to be about.");
      return;
    }

    var data = "email=" + encodeURIComponent(currentUserEmail);
    data += "&topic=" + encodeURIComponent(topic);
    data += "&quiz_request=" + encodeURIComponent(quizRequest);
    data += "&include_history=" + encodeURIComponent(includeHistory);

    fetch(api + "/quizzes", {
      method: "POST",
      body: data,
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Authorization: authorizationHeader(),
      },
    })
      .then(function (res) {
        if (!res.ok) {
          return res.text().then(function (txt) {
            throw new Error(txt || "Quiz request failed");
          });
        }
        return res.json();
      })
      .then(function (payload) {
        var quiz = null;
        try {
          quiz = JSON.parse(payload.quiz_json);
        } catch (e) {
          console.log("Failed to parse quiz_json:", e);
          quizContainer.innerHTML = "<p>Could not parse quiz from server.</p>";
          return;
        }

        renderQuiz(quiz, payload.quiz_id);

        // NEW: refresh quiz history list so the new quiz shows up
        loadQuizHistory();
      })
      .catch(function (err) {
        console.log("Quiz error:", err);
        quizContainer.innerHTML = "<p>⚠️ Error creating quiz.</p>";
      });
  };

  // NEW: when quiz UI is initialized and user is logged in, load history
  if (isLoggedIn()) {
    loadQuizHistory();
  }
}

/* quiz JSON format example we expect:
{
  "topic": "CS",
  "difficulty": "easy",
  "questions": [
    {
      "id": 1,
      "prompt": "What is a stack?",
      "options": ["LIFO structure", "FIFO structure", "Sorting algo", "Hash map"],
      "correct_index": 0
    },
    ...
  ]
}
*/

function loadQuizHistory() {
  if (!isLoggedIn()) {
    return;
  }

  var email = currentUserEmail || localStorage.getItem("email");
  if (!email) {
    return;
  }

  var data = "email=" + encodeURIComponent(email);
  data += "&limit=" + encodeURIComponent("10");

  fetch(api + "/quizzes/list", {
    method: "POST",
    body: data,
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Authorization: authorizationHeader(),
    },
  })
    .then(function (res) {
      if (!res.ok) {
        return res.text().then(function (txt) {
          console.log("Failed to load quiz history:", res.status, txt);
          throw new Error(txt || "history failed");
        });
      }
      return res.json();
    })
    .then(function (payload) {
      if (!payload || !payload.items) {
        return;
      }
      renderQuizList(payload.items);
    })
    .catch(function (err) {
      console.log("quiz history error:", err);
    });
}

function renderQuizList(items) {
  var quizListDiv = document.getElementById("quiz-list");
  if (!quizListDiv) {
    return;
  }

  quizListDiv.innerHTML = "";

  if (!items.length) {
    quizListDiv.innerHTML = "<p>No quizzes yet.</p>";
    return;
  }

  for (var i = 0; i < items.length; i++) {
    (function (quiz) {
      var btn = document.createElement("button");
      btn.className = "quiz-list-item";

      var label = "Quiz #" + quiz.id + " (" + quiz.topic + ")";
      if (quiz.status) {
        label += " — " + quiz.status;
      }
      if (quiz.score != null && quiz.status === "completed") {
        label += " — score: " + Number(quiz.score).toFixed(1) + "%";
      }

      btn.textContent = label;

      btn.onclick = function () {
        var quizObj = null;
        try {
          quizObj = JSON.parse(quiz.quiz_json);
        } catch (e) {
          console.log("Error parsing quiz_json for quiz", quiz.id, e);
          showMessage("Could not load this quiz", "error");
          return;
        }

        // reuse your existing renderer + submitQuiz logic
        renderQuiz(quizObj, quiz.id);
      };

      quizListDiv.appendChild(btn);
    })(items[i]);
  }
}

function renderQuiz(quiz, quiz_id) {
  var quizContainer = document.getElementById("quiz-container");
  if (!quizContainer) {
    return;
  }

  var html = "<h3>Your Quiz</h3>";
  html += "<form id='quiz-form'>";

  if (!quiz.questions || quiz.questions.length === 0) {
    quizContainer.innerHTML = "<p>No questions in quiz.</p>";
    return;
  }

  for (var i = 0; i < quiz.questions.length; i++) {
    var q = quiz.questions[i];
    html += "<div class='quiz-question'>";
    html += "<p><strong>" + (i + 1) + ". " + q.prompt + "</strong></p>";

    for (var j = 0; j < q.options.length; j++) {
      html +=
        "<label><input type='radio' name='q" +
        q.id +
        "' value='" +
        j +
        "'> " +
        q.options[j] +
        "</label><br>";
    }

    html += "</div>";
  }

  html += "<button type='submit'>Submit Quiz</button>";
  html += "</form>";

  quizContainer.innerHTML = html;

  var quizForm = document.getElementById("quiz-form");
  quizForm.onsubmit = function (event) {
    event.preventDefault();
    submitQuiz(quiz, quiz_id);
  };
}

function submitQuiz(quiz, quiz_id) {
  var answers = {};
  var correct = 0;

  for (var i = 0; i < quiz.questions.length; i++) {
    var q = quiz.questions[i];
    var selected = document.querySelector(
      "input[name='q" + q.id + "']:checked",
    );
    var choice = selected ? parseInt(selected.value) : -1;

    answers[q.id] = choice;
    if (typeof q.correct_index === "number" && choice === q.correct_index) {
      correct++;
    }
  }

  var score = 0;
  if (quiz.questions.length > 0) {
    score = (correct / quiz.questions.length) * 100;
  }

  var data = "quiz_id=" + encodeURIComponent(quiz_id);
  data += "&user_answers_json=" + encodeURIComponent(JSON.stringify(answers));
  data += "&score=" + encodeURIComponent(String(score));

  fetch(api + "/quizzes/complete", {
    method: "POST",
    body: data,
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Authorization: authorizationHeader(),
    },
  })
    .then(function (res) {
      if (!res.ok) {
        return res.text().then(function (txt) {
          throw new Error(txt || "Submit quiz failed");
        });
      }
      var quizContainer = document.getElementById("quiz-container");
      quizContainer.innerHTML =
        "<h3>Quiz Complete!</h3><p>Your score: " + score.toFixed(1) + "%</p>";
    })
    .catch(function (err) {
      console.log("Submit quiz error:", err);
      var quizContainer = document.getElementById("quiz-container");
      quizContainer.innerHTML =
        "<p>⚠️ Error submitting quiz. Please try again.</p>";
    });
}

/* ---------- INIT ---------- */

window.addEventListener("load", function () {
  initAuthForm();
  initChatUI();
  initModeToggle();
  initQuizUI();
  if (isLoggedIn()) {
    let storedEmail = localStorage.getItem("email");
    if (storedEmail) {
      showChatForUser(storedEmail);
    }
  }

  check();
});
