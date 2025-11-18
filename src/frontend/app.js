// authurization header neccesary when we want to authourize someone to do something.

//  send it with the fetch
let api = "http://localhost:5000";
function authorizationHeader() {
  // store a cookie is in the local storage
  let sessionID = localStorage.getItem("sessionID");
  if (sessionID) {
    console.log("Found a session id in authheader");
    // if it does exist make the header be this
    return "Bearer " + sessionID;
  } else {
    // if not then its null
    return null;
  }
}

// when they visit the page we want to make a session id

function createSessionID() {
  // vists the sessions endpoint and sends the header
  fetch(api + "/sessions", {
    headers: {
      Authorization: authorizationHeader(),
    },
  }).then(function (response) {
    if (response.status == 200) {
      response.json().then(function (session) {
        localStorage.setItem("sessionID", session.id);
        // can add logic to prefrences
      });
    }
  });
}
