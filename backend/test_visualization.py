"""
Test script for verifying WebSocket connection and visualization
"""
import requests
import time
import sys
import os

def test_socket_connection():
    """Test the WebSocket connection by calling the test endpoint"""
    print("Testing WebSocket connection...")
    try:
        response = requests.get('http://localhost:5000/api/test/socket')
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print("✅ Test frame emitted successfully")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_training_start():
    """Test starting the training process"""
    print("Testing training start...")
    try:
        response = requests.post(
            'http://localhost:5000/api/training/start',
            json={
                'episodes': 10,  # Small number for testing
                'batch_size': 32
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print("✅ Training started successfully")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_training_status():
    """Test checking the training status"""
    print("Testing training status...")
    try:
        response = requests.get('http://localhost:5000/api/training/status')
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print(f"✅ Training status: {data.get('is_training', False)}")
            return data.get('is_training', False)
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_training_stop():
    """Test stopping the training process"""
    print("Testing training stop...")
    try:
        response = requests.post('http://localhost:5000/api/training/stop')
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print("✅ Training stopped successfully")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def run_all_tests():
    """Run all tests in sequence"""
    print("=== Running Visualization Tests ===")
    
    # Test socket connection
    socket_test = test_socket_connection()
    if not socket_test:
        print("⚠️ Socket test failed, but continuing with other tests")
    
    # Test training start
    training_start = test_training_start()
    if not training_start:
        print("❌ Training start failed, aborting remaining tests")
        return
    
    # Wait a moment for training to initialize
    print("Waiting for training to initialize...")
    time.sleep(5)
    
    # Check training status
    is_training = test_training_status()
    if not is_training:
        print("⚠️ Training status check failed or training not running")
    
    # If training is running, let it run for a bit
    if is_training:
        print("Training is running. Letting it run for 10 seconds...")
        time.sleep(10)
    
    # Stop training
    training_stop = test_training_stop()
    if not training_stop:
        print("⚠️ Training stop failed")
    
    # Final status check
    time.sleep(2)
    final_status = test_training_status()
    
    print("\n=== Test Summary ===")
    print(f"Socket Connection: {'✅ Passed' if socket_test else '❌ Failed'}")
    print(f"Training Start: {'✅ Passed' if training_start else '❌ Failed'}")
    print(f"Training Status: {'✅ Passed' if is_training else '⚠️ Warning'}")
    print(f"Training Stop: {'✅ Passed' if training_stop else '⚠️ Warning'}")
    print(f"Final Status: {'✅ Stopped' if not final_status else '⚠️ Still Running'}")

if __name__ == "__main__":
    run_all_tests() 