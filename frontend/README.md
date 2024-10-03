### Set up Frontend
  
1. Change the current working directory to "frontend"
  
  ```  
  cd frontend
  ```

2. Install Dependencies:
   
  ```bash
  npm install
  ```

3. Set Up Firebase:
   - Create a Firebase project at [Firebase Console](https://console.firebase.google.com/).
   - Obtain your Firebase config credentials.
   - Add your Firebase config to `src/firebase/firebaseConfig.js`.

4. Start the Development Server:
   
  ```bash
  npm start
  ```

   This will run the React development server. You can view the website at `http://localhost:3000`.

### Folder Structure

The project folder structure is organized as follows:

- **`public/`**: Contains static assets and the main `index.html` file.
- **`src/`**: Contains all the source code for the React.js frontend.
  - **`assets/`**: Static assets like images, fonts, etc.
  - **`components/`**: Reusable components
    - **`Header/`**: Header component
    - **`Helmet/`**: Helmet component
    - **`Layout/`**: Layout components
      - `Modal.jsx`: Modal component
      - `PatientForm.jsx`: PatientForm component
      - `PatientInfo.jsx`: PatientInfo component
  - **`custom-hooks/`**: Custom React hooks
  - **`pages/`**: Pages of the application
    - `ChatBot.jsx`: ChatBot page component
    - `Home.jsx`: Home page component
    - `Login.jsx`: Login page component
    - `Signup.jsx`: Signup page component
  - **`redux/`**: Redux setup
  - **`routers/`**: Router setup
  - **`styles/`**: CSS styles
    - `App.css`: Global styles
  - `App.js`: Main application component
  - `firebase.config.js`: Firebase configuration
  - `index.js`: Entry point