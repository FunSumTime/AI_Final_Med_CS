import os,base64

# our class to make cookies want to give one when they log in
class SessionStore:
    def __init__(self):
        self.session_data= {}

# every time someone comes to the site we generate a session for them
    def generate_session_id(self):
        # get the random string
        rnum = os.urandom(32)
        # encode base64 then decode using utf-8
        encoded = base64.b64encode(rnum).decode("utf-8")
        print(encoded)
        return encoded


    def create_session(self):
        session_id = self.generate_session_id()
        # generate random number and keep a dictionary that will keep track of that random number
        #  {number: {"color":blue} the inner dictionary is tracking there session} 
        self.session_data[session_id] = {}
        return session_id
    
# retrive session data
    def get_session_data(self,session_id):
        if session_id in self.session_data:
            # if in the dictionary give them the data
            return self.session_data[session_id]
        else:
            return None
    
if __name__ == "__main__":
    s = SessionStore()
    my_id = s.generate_session_id()