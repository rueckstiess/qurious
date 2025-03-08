import random

from pymongo import ASCENDING, IndexModel, MongoClient

from ..environment import Environment


class MongoDBEnvironment(Environment):
    def __init__(self, mongo_uri, db_name: str, coll_name: str):
        """
        Initialize the MongoDB environment.

        Args:
            mongo_uri (str): The MongoDB URI for connecting to the database.
            database_name (str): The name of the database to use.
        """
        super().__init__()

        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.coll_name = coll_name

        # create client
        self.client = MongoClient(mongo_uri)
        self.database = self.client[db_name]
        self.collection = self.database[coll_name]

        # indexes
        self.indexes = self._get_indexes()

        # query execution cache
        self.query_cache = {}

    def _get_indexes(self):
        """
        Get the indexes of the MongoDB collection.

        Returns:
            list: The indexes of the MongoDB collection.
        """
        index_info = self.collection.index_information()
        indexes = {name: spec["key"] for name, spec in index_info.items() if name != "_id_"}
        indexes["COLL_SCAN"] = [("$natural", 1)]  # Add collection scan as a hint
        return indexes

    def _create_2d_dataset(self, num_documents: int, x_range: tuple = (0, 100), y_range: tuple = (0, 100)):
        """
        Create a 2D dataset in the MongoDB collection.

        Args:
            num_documents (int): The number of documents to create.
        """
        self.collection.drop()  # Clear the collection before inserting new documents
        self.collection.drop_indexes()

        def create_random_doc(i):
            x = random.randint(x_range[0], x_range[1])
            y = random.randint(y_range[0], y_range[1])
            return {"_id": i, "x": x, "y": y}

        # Insert random documents into the collection
        self.collection.insert_many(create_random_doc(i) for i in range(num_documents))

        self.indexes = self.collection.create_indexes(
            [
                IndexModel([("x", ASCENDING)]),
                IndexModel([("y", ASCENDING)]),
                IndexModel([("x", ASCENDING), ("y", ASCENDING)]),
            ]
        )

        self.indexes = self._get_indexes()  # Update indexes after creation

    def _create_random_query(self):
        x_min, x_max = sorted([random.randint(0, 100), random.randint(0, 100)])
        y_min, y_max = sorted([random.randint(0, 100), random.randint(0, 100)])
        self._state = {"x": {"$gte": x_min, "$lte": x_max}, "y": {"$gte": y_min, "$lte": y_max}}

        return self._state

    def step(self, action):
        """
        Execute a query in the MongoDB collection and return the execution
        time as negative reward. The action is the hint provided to the
        MongoDB query.

        Args:
            action: The index to hint with (None for collection scan)

        Returns:
            next_state: The new state after taking the action
            reward: The negative execution time
            done: Whether the episode has terminated
            info: Additional information (optional)
        """

        # get query
        query = self.get_state()

        # get hint
        hint = self.indexes[action]

        # execute query with explain
        query_str = str(query)
        explain = None
        
        if query_str in self.query_cache:
            reward = self.query_cache[query_str]
        else:
            explain = self.collection.find(query, hint=hint).explain()["executionStats"]
            reward = -explain["executionTimeMillis"]
            self.query_cache[query_str] = reward

        self._done = True
        next_state = self._create_random_query()  # Create a new random query
        
        # If explain is None (cached query), fetch recent execution stats
        if explain is None:
            # For cached queries, we might want to provide the most recent stats
            # or default stats. Here we're looking for the most recent explain result
            # from the client to maintain consistency.
            explain = self.collection.find({}, limit=1).explain()["executionStats"]
            
        info = {"explain": explain}

        return next_state, reward, self._done, info

    @property
    def action_space(self):
        """
        Action space is the indexes of the MongoDB collection, plus None,
        which represents a collection scan.

        Returns:
            The action space
        """
        return list(self.indexes.keys())
