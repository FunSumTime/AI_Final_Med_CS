# CS/Med AI Tutor

## Overview

CS/Med AI Tutor is an AI tutoring system built to help users study computer science and medical-related topics. The goal of the project was to create more than just a basic chatbot. Instead, I wanted to build a tutor that could answer questions, generate quizzes, track how the user performs, and grow its knowledge base over time.

The project uses a RAG pipeline, which allows the tutor to retrieve information from stored knowledge instead of only relying on the model itself. Computer science and medical topics each have their own knowledge base, so the tutor can follow different paths depending on what the user is asking about.

## Project Purpose

This project was created for my AI Agents class and was also used as a final project for my Web Applications class. I wanted to build something that connected to my long-term goal of combining computer science and medicine.

The main idea was to make a tutor that could help a user learn over time. As the user asks questions and takes quizzes, the system can store useful information and track what areas the user may be struggling with.

## Features

- AI tutor for computer science and medical topics
- Separate RAG knowledge bases for CS and Med content
- Question-answering using stored material
- Web search fallback when information is not already stored
- Quiz generation for users
- Tracks quiz results and user performance
- Helps identify weak areas based on quiz results
- Login/authentication system
- User mapping so the tutor knows who it is helping
- Knowledge base that can grow over time

## How It Works

The tutor follows a general loop:

1. The user asks a question or requests help.
2. The system checks whether the topic is related to computer science or medicine.
3. Based on the topic, the tutor searches the correct knowledge base.
4. If the information is found, the tutor uses it to answer the user.
5. If the information is not found, the system can search the web for more information.
6. Useful information can be stored so the tutor can answer similar questions faster in the future.
7. The tutor can generate quizzes and use quiz results to see where the user may need more practice.

## Why RAG Was Used

RAG, or Retrieval-Augmented Generation, was used so the tutor could answer questions using information from a knowledge base instead of only depending on the AI model’s general training.

This made the project more useful because the tutor could:

- Pull from specific study material
- Keep CS and Med information separated
- Add new information over time
- Give more grounded answers
- Grow with the user as they continue learning

## Web Application Side

Since this project was also used for my Web Applications final, I added web app features such as login authentication and user tracking. This allowed the system to know which user was interacting with the tutor and connect quiz results or stored information back to that user.

This helped me practice building a full system that connected the frontend, backend, database, and AI agent together.

## What I Learned

Through this project, I learned more about how AI agents can be used for education. I also got more experience working with RAG pipelines, databases, authentication, and web app structure.

One of the biggest things I learned was that an AI tutor should not only give answers. It should also be able to check understanding, track progress, and adjust based on what the user is struggling with.

## Future Improvements

Some future improvements I would like to add include:

- Better quiz analysis
- More detailed user progress tracking
- A cleaner dashboard for weak topics
- More advanced CS and Med knowledge bases
- Better filtering for web-searched information
- More personalized study paths
- Support for uploaded notes or PDFs

## Short Description

A CS/Med AI tutoring system that uses separate RAG knowledge bases for computer science and medical topics. The tutor can answer questions, generate quizzes, track user performance, and identify areas where the user may need more help. Its knowledge base grows over time, allowing the system to improve alongside the user.
