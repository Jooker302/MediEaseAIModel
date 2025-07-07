# MediEase Python Model

## Overview
MediEase Python Model is the backend component of the MediEase healthcare system, responsible for processing health reports and generating personalized recommendations for medicine, exercises, and food plans. Built with Python, it integrates with a MongoDB database and serves requests from the Next.js website and React Native app. Developed as a Final Year Project (FYP), this prototype has limited accuracy due to constrained datasets.

## Features
- Processes health report data (e.g., blood pressure, sugar levels)
- Generates three types of recommendations:
  - Medicine suggestions
  - Exercise plans
  - Food plans
- REST API endpoints for integration with the website and app
- MongoDB for storing model data and recommendations
> **Note**: The recommendation model has limited accuracy due to small and incomplete datasets.

## Tech Stack
- **Language**: Python
- **Framework**: Flask (or specify if FastAPI, Django, etc., was used)
- **Database**: MongoDB
- **Libraries**: (e.g., pandas, scikit-learn, pymongo, etc.)
- **Model**: Custom-trained machine learning model (or rule-based system)

## Related Repositories
- [MediEase Website](https://github.com/Jooker302/MediEaseWebsite): Next.js Website
- [MediEase Mobile App](https://github.com/Jooker302/MediEaseUserApp): React Native App

## Prerequisites
- Python (v3.8 or higher)
- MongoDB (local or cloud instance, e.g., MongoDB Atlas)
- pip for installing dependencies
- Next.js backend running (for API integration)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url-model>
   cd mediease-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory and add:
     ```env
     MONGODB_URI=<your-mongodb-connection-string>
     ```

4. Run the server:
   ```bash
   python app.py
   ```
   The server will run on [http://localhost:5000](http://localhost:5000) (or your specified port).

## Usage
- Send a POST request to the recommendation endpoint (e.g., `/api/recommend`) with health report data (e.g., blood pressure, sugar levels).
- The model processes the data and returns:
  - Recommended medicines
  - Suggested exercises
  - Food plan
- Ensure the MongoDB instance and Next.js backend are running for full functionality.

## Limitations
- Recommendation model accuracy is low due to limited and incomplete datasets
- Prototype-level implementation for FYP purposes
- Basic error handling and model validation

## Contributing
This project was created for educational purposes. Feel free to fork and experiment, but it is not actively maintained.

## License
[MIT License](LICENSE)
