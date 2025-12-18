# Cyber-Bullying-Thesis-
An AI-powered system for detecting cyberbullying across English, Bengali, and Banglish (code-mixed) text. Features a custom language identifier for routing inputs to language-specific models including BERT, BiLSTM, and SVM to achieve high-accuracy, real-time detection.

**Core Technologies**
Models: BERT, mBERT, BiLSTM, SVM, Random Forest.
Backend: FastAPI for low-latency API routing.
Data Sources: Sourced from Kaggle and Mendeley Data (Bengali Hate Speech, Cyberbullying Classification, and Banglish datasets).

**Results Summary**
English: SVM reached ~89.9% accuracy, while BERT provided superior contextual understanding.
Bengali: Achieved high performance using specialized tools like BanglaBERT.
Banglish: Successfully addressed the lack of standardized spelling through specialized character-level n-gram and model tuning.
