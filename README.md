
# Banking Chatbot Web Application

This project is a web application for a banking chatbot system, providing user-friendly interfaces for login and interaction with a chatbot for banking assistance. The backend processes and preprocesses data to serve as a robust system for banking insights.

---

## Features
1. **Login Page**: 
   - User authentication interface for customers.
   - Minimalistic and responsive design.

2. **Chatbot Interface**: 
   - Interactive FAQ assistant powered by MongoDB Atlas and IBM Watsonx.
   - Chatbot widget for real-time banking assistance.

3. **Backend Processing**:
   - Data preprocessing script (`preprocessing.py`) to clean and prepare data.
   - Core logic processing script (`processing.py`) to manage requests and responses.

---

## Files

### 1. `login.html`
- A responsive login page with a clean design.
- Accepts **Customer ID** for authentication.
- Includes an integrated chatbot widget for quick FAQs assistance.

### 2. `chatbot.html`
- A full-fledged chatbot interface with a sleek design.
- Includes welcome messages, logout functionality, and real-time interaction.
- Backend integration for fetching data via APIs.

### 3. `preprocessing.py`
- Prepares and cleans the data to ensure efficient processing.
- Key tasks: Data normalization, handling missing values, and optimization for MongoDB.

### 4. `processing.py`
- Implements the core chatbot logic.
- Handles API calls, processes user queries, and fetches responses from the database or AI model.

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- MongoDB
- Node.js (for serving HTML files if required)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing script to prepare data:
   ```bash
   python preprocessing.py
   ```

4. Start the application backend:
   ```bash
   python processing.py
   ```

5. Serve the frontend:
   - Use a web server like Flask, Django, or any static file server to host the HTML files.

---

## Usage
1. Open `login.html` in a web browser.
2. Enter **Customer ID** and log in.
3. Interact with the chatbot for banking assistance.

---

## Screenshots
<img width="1512" alt="Screenshot 2025-01-07 at 6 11 41 PM" src="https://github.com/user-attachments/assets/c3a9883f-59fa-443c-b7c4-f3bbd16f008d" />
<img width="1512" alt="Screenshot 2025-01-07 at 6 14 11 PM" src="https://github.com/user-attachments/assets/4de6b30c-704b-4972-89d2-991b4f7453bf" />



---

## Requirements

Refer to `requirements.txt` for the list of dependencies.

---

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, MongoDB
- **APIs**: IBM Watsonx AI
- **Database**: MongoDB Atlas
