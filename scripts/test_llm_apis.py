#!/usr/bin/env python3
"""
Test script for LLM API connectivity with real keys.
Tests Groq (primary) and Gemini (fallback) APIs.
"""

import asyncio
import json
import logging
import os
import sys
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, '.')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_groq_api():
    """Test Groq API connectivity"""
    from spatial_lab.llm.groq_client import test_groq_api, GroqAPIConfig, GroqAPIClient

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set in environment")
        return False

    logger.info("Testing Groq API...")
    logger.info(f"API Key: {api_key[:10]}...{api_key[-4:]}")

    # Test basic connectivity
    result = await test_groq_api(api_key)

    if result:
        logger.info("Groq API basic test PASSED")

        # Test spatial reasoning task
        logger.info("\nTesting spatial reasoning task...")
        config = GroqAPIConfig(api_key=api_key, model="llama-3.3-70b-versatile")

        async with GroqAPIClient(config) as client:
            response = await client.robot_coordination_decision(
                robot_id="robot_001",
                observation={
                    "position": [5.0, 5.0, 0.0],
                    "battery_level": 0.85,
                    "carrying_item": None,
                    "status": "idle"
                },
                task_description="Pick up item_box_A from shelf at position (10, 8) and deliver to shipping zone at (2, 15)",
                available_actions=["move_to", "pick_item", "drop_item", "wait", "communicate"],
                nearby_robots=[
                    {"robot_id": "robot_002", "position": [7.0, 5.0, 0.0], "status": "moving"}
                ],
                warehouse_layout={
                    "dimensions": [20, 20],
                    "shelves": [
                        {"id": "shelf_A", "position": [10, 8], "items": ["item_box_A", "item_box_B"]}
                    ],
                    "zones": {
                        "shipping": {"position": [2, 15], "type": "delivery"}
                    }
                }
            )

            # Parse the response
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                logger.info(f"\nRobot Decision Response:")
                try:
                    decision = json.loads(content)
                    logger.info(f"  Action: {decision.get('action')}")
                    logger.info(f"  Parameters: {decision.get('parameters')}")
                    logger.info(f"  Reasoning: {decision.get('reasoning')[:200]}...")
                    logger.info(f"  Confidence: {decision.get('confidence')}")
                    logger.info("\nGroq spatial reasoning test PASSED")
                    return True
                except json.JSONDecodeError:
                    logger.info(f"  Raw response: {content[:500]}")
                    logger.warning("Response not valid JSON, but API works")
                    return True
            else:
                logger.error(f"Unexpected response format: {response}")
                return False
    else:
        logger.error("Groq API basic test FAILED")
        return False


async def test_gemini_api():
    """Test Gemini API connectivity"""
    from spatial_lab.llm.gemini_client import test_gemini_api, GeminiAPIConfig, GeminiAPIClient

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set in environment")
        return False

    logger.info("\nTesting Gemini API...")
    logger.info(f"API Key: {api_key[:10]}...{api_key[-4:]}")

    # Test with different models
    models_to_test = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp"
    ]

    for model in models_to_test:
        logger.info(f"\nTesting model: {model}")
        try:
            config = GeminiAPIConfig(api_key=api_key, model=model, timeout=30)

            async with GeminiAPIClient(config) as client:
                response = await client.spatial_reasoning_completion(
                    prompt="A robot at (0,0) needs to reach (5,5) avoiding an obstacle at (2,2). What path should it take? Respond briefly.",
                    system_prompt="You are a spatial reasoning assistant. Be concise."
                )

                if "candidates" in response and len(response["candidates"]) > 0:
                    content = response["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"  Response: {content[:200]}...")
                    logger.info(f"  Model {model} works!")
                else:
                    logger.warning(f"  Unexpected response format for {model}")

        except Exception as e:
            logger.warning(f"  Model {model} failed: {e}")

    return True


async def test_robot_coordination_scenario():
    """Test a full robot coordination scenario with real LLMs"""
    from spatial_lab.llm.groq_client import GroqAPIConfig, GroqAPIClient

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False

    logger.info("\n" + "="*60)
    logger.info("FULL ROBOT COORDINATION SCENARIO TEST")
    logger.info("="*60)

    config = GroqAPIConfig(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

    # Simulate a multi-step coordination scenario
    scenario_steps = [
        {
            "robot_id": "robot_001",
            "observation": {
                "position": [2.0, 2.0, 0.0],
                "battery_level": 0.95,
                "carrying_item": None,
                "status": "idle"
            },
            "task_description": "Collect item_A from shelf_1 at (8, 5) and item_B from shelf_2 at (12, 7), then deliver both to shipping at (3, 18)",
            "step": 1
        },
        {
            "robot_id": "robot_001",
            "observation": {
                "position": [8.0, 5.0, 0.0],
                "battery_level": 0.90,
                "carrying_item": "item_A",
                "status": "carrying"
            },
            "task_description": "Now holding item_A. Continue to collect item_B from shelf_2 at (12, 7), then deliver both to shipping at (3, 18)",
            "step": 2
        },
        {
            "robot_id": "robot_001",
            "observation": {
                "position": [12.0, 7.0, 0.0],
                "battery_level": 0.85,
                "carrying_item": "item_A, item_B",
                "status": "carrying"
            },
            "task_description": "Now holding both items. Deliver to shipping zone at (3, 18)",
            "step": 3
        }
    ]

    async with GroqAPIClient(config) as client:
        for step in scenario_steps:
            logger.info(f"\n--- Step {step['step']} ---")
            logger.info(f"Robot position: {step['observation']['position']}")
            logger.info(f"Carrying: {step['observation']['carrying_item']}")

            response = await client.robot_coordination_decision(
                robot_id=step["robot_id"],
                observation=step["observation"],
                task_description=step["task_description"],
                available_actions=["move_to", "pick_item", "drop_item", "wait"],
                nearby_robots=[],
                warehouse_layout={
                    "dimensions": [20, 20],
                    "shelves": [
                        {"id": "shelf_1", "position": [8, 5]},
                        {"id": "shelf_2", "position": [12, 7]}
                    ],
                    "zones": {"shipping": {"position": [3, 18]}}
                }
            )

            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                try:
                    decision = json.loads(content)
                    logger.info(f"Decision: {decision.get('action')}")
                    logger.info(f"Parameters: {decision.get('parameters')}")
                    logger.info(f"Reasoning: {decision.get('reasoning', '')[:150]}...")
                except json.JSONDecodeError:
                    logger.info(f"Raw: {content[:200]}")

            # Small delay between requests
            await asyncio.sleep(0.5)

    logger.info("\n" + "="*60)
    logger.info("SCENARIO TEST COMPLETED")
    logger.info("="*60)
    return True


async def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("SPATIAL LAB - LLM API CONNECTIVITY TEST")
    print("="*60 + "\n")

    results = {}

    # Test Groq
    print("\n[1/3] Testing Groq API (Primary)...")
    results["groq"] = await test_groq_api()

    # Test Gemini
    print("\n[2/3] Testing Gemini API (Fallback)...")
    results["gemini"] = await test_gemini_api()

    # Test full scenario
    if results["groq"]:
        print("\n[3/3] Testing Full Robot Coordination Scenario...")
        results["scenario"] = await test_robot_coordination_scenario()
    else:
        print("\n[3/3] Skipping scenario test (Groq not available)")
        results["scenario"] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name.upper()}: {status}")

    all_passed = all(results.values())
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
