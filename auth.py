import sqlite3
import hashlib
import streamlit as st

class AuthManager:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path

    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, email, password):
        """Create a new user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            hashed_password = self.hash_password(password)

            cursor.execute("""
                INSERT INTO users (email, password)
                VALUES (?, ?)
            """, (email, hashed_password))

            conn.commit()
            conn.close()
            return True, "Account created successfully!"
        except sqlite3.IntegrityError:
            return False, "Email already exists!"
        except Exception as e:
            return False, f"Error creating account: {str(e)}"

    def authenticate_user(self, email, password):
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            hashed_password = self.hash_password(password)

            cursor.execute("""
                SELECT id FROM users
                WHERE email = ? AND password = ?
            """, (email, hashed_password))

            result = cursor.fetchone()
            conn.close()

            if result:
                return True, result[0]  # Return True and user ID
            else:
                return False, "Invalid email or password"
        except Exception as e:
            return False, f"Authentication error: {str(e)}"

    def save_result(self, email, resume_name, score):
        """Save analysis result for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO results (email, resume_name, score)
                VALUES (?, ?, ?)
            """, (email, resume_name, score))

            conn.commit()
            conn.close()
            return True, "Result saved successfully!"
        except Exception as e:
            return False, f"Error saving result: {str(e)}"

    def get_user_results(self, email):
        """Get all results for a specific user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT resume_name, score
                FROM results
                WHERE email = ?
                ORDER BY score DESC
            """, (email,))

            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            return []

    def clear_user_results(self, email):
        """Clear all results for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM results WHERE email = ?", (email,))
            conn.commit()
            conn.close()
            return True, "Results cleared successfully!"
        except Exception as e:
            return False, f"Error clearing results: {str(e)}"