#!/usr/bin/env python3
"""
Test script for the authentication system
"""

from auth import AuthManager

def test_auth():
    auth = AuthManager()

    print("🧪 Testing Authentication System...")

    # Test user creation
    print("\n1. Testing user creation...")
    success, message = auth.create_user("test@example.com", "password123")
    print(f"   Result: {success} - {message}")

    # Test duplicate user
    print("\n2. Testing duplicate user...")
    success, message = auth.create_user("test@example.com", "password123")
    print(f"   Result: {success} - {message}")

    # Test authentication
    print("\n3. Testing authentication...")
    success, user_id = auth.authenticate_user("test@example.com", "password123")
    print(f"   Result: {success} - User ID: {user_id}")

    # Test wrong password
    print("\n4. Testing wrong password...")
    success, message = auth.authenticate_user("test@example.com", "wrongpass")
    print(f"   Result: {success} - {message}")

    # Test saving results
    print("\n5. Testing result saving...")
    success, message = auth.save_result("test@example.com", "resume1.pdf", 85.5)
    print(f"   Result: {success} - {message}")

    success, message = auth.save_result("test@example.com", "resume2.pdf", 92.3)
    print(f"   Result: {success} - {message}")

    # Test retrieving results
    print("\n6. Testing result retrieval...")
    results = auth.get_user_results("test@example.com")
    print(f"   Results: {results}")

    print("\n✅ Authentication system test completed!")

if __name__ == "__main__":
    test_auth()