import unittest
from unittest.mock import MagicMock, patch

from qurious.rl.environments.mongodb.mongodb_environment import MongoDBEnvironment


class TestMongoDBEnvironment(unittest.TestCase):
    """Test the MongoDBEnvironment class."""

    def setUp(self):
        """Set up test environment with mocked MongoDB."""
        # Set up patches for MongoDB-related components
        self.mongo_client_patch = patch("qurious.rl.environments.mongodb.mongodb_environment.MongoClient")
        self.mock_mongo_client = self.mongo_client_patch.start()

        # Setup mock collection, database and client
        self.mock_collection = MagicMock()
        self.mock_database = MagicMock()
        self.mock_client = MagicMock()

        # Configure mocks
        self.mock_client.__getitem__.return_value = self.mock_database
        self.mock_database.__getitem__.return_value = self.mock_collection
        self.mock_mongo_client.return_value = self.mock_client

        # Set up index information
        self.mock_index_info = {
            "_id_": {"key": [("_id", 1)]},
            "x_1": {"key": [("x", 1)]},
            "y_1": {"key": [("y", 1)]},
            "x_1_y_1": {"key": [("x", 1), ("y", 1)]},
        }
        self.mock_collection.index_information.return_value = self.mock_index_info

        # Create a test environment
        self.mongodb_env = MongoDBEnvironment("mongodb://localhost:27017", "test_db", "test_collection")

        # Mock explain result for queries
        self.mock_explain_result = {
            "executionStats": {"executionTimeMillis": 10, "nReturned": 5, "totalDocsExamined": 100}
        }
        self.mock_cursor = MagicMock()
        self.mock_cursor.explain.return_value = self.mock_explain_result
        self.mock_collection.find.return_value = self.mock_cursor

    def tearDown(self):
        """Clean up patches."""
        self.mongo_client_patch.stop()

    def test_init(self):
        """Test environment initialization."""
        self.assertEqual(self.mongodb_env.mongo_uri, "mongodb://localhost:27017")
        self.assertEqual(self.mongodb_env.db_name, "test_db")
        self.assertEqual(self.mongodb_env.coll_name, "test_collection")

        # Check MongoDB client was initialized
        self.mock_mongo_client.assert_called_once_with("mongodb://localhost:27017")

        # Check indexes were retrieved
        self.mock_collection.index_information.assert_called_once()

        # Check expected indexes (including the COLL_SCAN)
        self.assertIn("x_1", self.mongodb_env.indexes)
        self.assertIn("y_1", self.mongodb_env.indexes)
        self.assertIn("x_1_y_1", self.mongodb_env.indexes)
        self.assertIn("COLL_SCAN", self.mongodb_env.indexes)

    def test_get_indexes(self):
        """Test _get_indexes method."""
        indexes = self.mongodb_env._get_indexes()

        # Should have the 3 MongoDB indexes plus the COLL_SCAN
        self.assertEqual(len(indexes), 4)
        self.assertEqual(indexes["x_1"], [("x", 1)])
        self.assertEqual(indexes["y_1"], [("y", 1)])
        self.assertEqual(indexes["x_1_y_1"], [("x", 1), ("y", 1)])
        self.assertEqual(indexes["COLL_SCAN"], [("$natural", 1)])

    def test_create_2d_dataset(self):
        """Test _create_2d_dataset method."""
        # Track insert_many calls to check document structure
        self.mock_collection.insert_many = MagicMock()

        # Call the method
        self.mongodb_env._create_2d_dataset(num_documents=10, x_range=(1, 10), y_range=(5, 15))

        # Check that collection was cleared
        self.mock_collection.drop.assert_called_once()
        self.mock_collection.drop_indexes.assert_called_once()

        # Check that documents were inserted
        self.mock_collection.insert_many.assert_called_once()

        # Capture the documents passed to insert_many and verify their structure
        # We need to check the inner function create_random_doc is executed correctly
        call_args = self.mock_collection.insert_many.call_args[0][0]
        docs_list = list(call_args)  # Convert generator to list

        # Check documents structure and x,y values within specified ranges
        for i, doc in enumerate(docs_list):
            self.assertEqual(doc["_id"], i)
            self.assertIn("x", doc)
            self.assertIn("y", doc)
            self.assertGreaterEqual(doc["x"], 1)
            self.assertLessEqual(doc["x"], 10)
            self.assertGreaterEqual(doc["y"], 5)
            self.assertLessEqual(doc["y"], 15)

        # Check that indexes were created
        self.mock_collection.create_indexes.assert_called_once()

        # Check that indexes were reloaded
        self.assertEqual(self.mock_collection.index_information.call_count, 2)

    def test_create_random_query(self):
        """Test _create_random_query method."""
        query = self.mongodb_env._create_random_query()

        # Query should be a dictionary with 'x' and 'y' keys
        self.assertIsInstance(query, dict)
        self.assertIn("x", query)
        self.assertIn("y", query)

        # Each key should have $gte and $lte operators
        self.assertIn("$gte", query["x"])
        self.assertIn("$lte", query["x"])
        self.assertIn("$gte", query["y"])
        self.assertIn("$lte", query["y"])

        # $gte values should be less than or equal to $lte values
        self.assertLessEqual(query["x"]["$gte"], query["x"]["$lte"])
        self.assertLessEqual(query["y"]["$gte"], query["y"]["$lte"])

    def test_step_new_query(self):
        """Test step method with a new query."""
        # Create a fixed test query for consistent testing
        test_query = {"x": {"$gte": 10, "$lte": 50}, "y": {"$gte": 20, "$lte": 60}}
        self.mongodb_env._state = test_query

        # Make get_state return our test query
        original_get_state = self.mongodb_env.get_state
        self.mongodb_env.get_state = MagicMock(return_value=test_query)

        # Take a step with one of the indexes
        next_state, reward, done, info = self.mongodb_env.step("x_1")

        # Check the query was executed with the correct hint
        self.mock_collection.find.assert_called_once()
        self.mock_collection.find.assert_called_with(test_query, hint=[("x", 1)])

        # Check that the reward is negative execution time
        self.assertEqual(reward, -10)

        # Check that done flag is set
        self.assertTrue(done)

        # Check that next_state is a new random query
        self.assertIsInstance(next_state, dict)
        self.assertIn("x", next_state)
        self.assertIn("y", next_state)

        # Check info contains explain results
        self.assertEqual(info["explain"], self.mock_explain_result["executionStats"])

        # Restore original get_state
        self.mongodb_env.get_state = original_get_state

    def test_step_cached_query(self):
        """Test step method with a cached query."""
        # Create a random query state
        query = self.mongodb_env._create_random_query()
        original_query = query.copy()  # Make a copy to use in assertions

        # Add to cache
        self.mongodb_env.query_cache[str(original_query)] = -5

        # Make the get_state return our known query for consistent testing
        self.mongodb_env.get_state = MagicMock(return_value=original_query)

        # We need to reset the mock to clear the call count from previous tests
        self.mock_collection.find.reset_mock()

        # Mock the explain method for the find result
        mock_find_result = MagicMock()
        mock_find_result.explain.return_value = {
            "executionStats": {"executionTimeMillis": 5, "nReturned": 10, "totalDocsExamined": 50}
        }
        self.mock_collection.find.return_value = mock_find_result

        # Take a step
        next_state, reward, done, info = self.mongodb_env.step("y_1")

        # We now expect find to be called once for the fallback explain
        # but not with our original query
        self.mock_collection.find.assert_called_once()
        self.mock_collection.find.assert_called_with({}, limit=1)

        # Check that the reward is the cached value
        self.assertEqual(reward, -5)

        # Check that done flag is set
        self.assertTrue(done)

    def test_action_space(self):
        """Test action_space property."""
        action_space = self.mongodb_env.action_space

        # Should have all index names
        self.assertIn("x_1", action_space)
        self.assertIn("y_1", action_space)
        self.assertIn("x_1_y_1", action_space)
        self.assertIn("COLL_SCAN", action_space)

        self.assertEqual(len(action_space), 4)

    def test_observation_space(self):
        """Test observation_space property raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            _ = self.mongodb_env.observation_space

    def test_reset(self):
        """Test reset method."""
        # Set the done flag to True
        self.mongodb_env._done = True

        # Call reset
        state = self.mongodb_env.reset()

        # Check that done flag is reset
        self.assertFalse(self.mongodb_env.done)

        # Check that state is returned
        self.assertEqual(state, self.mongodb_env.get_state())


if __name__ == "__main__":
    unittest.main()
