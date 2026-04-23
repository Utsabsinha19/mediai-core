from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from backend.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    database: AsyncIOMotorDatabase = None

    @classmethod
    async def connect(cls):
        """Connect to MongoDB"""
        try:
            cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
            cls.database = cls.client[settings.DATABASE_NAME]
            
            # Test connection
            await cls.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB: {settings.DATABASE_NAME}")
            
            # Create indexes
            await cls.create_indexes()
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise

    @classmethod
    async def close(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            logger.info("✅ MongoDB connection closed")

    @classmethod
    async def create_indexes(cls):
        """Create database indexes for performance"""
        try:
            # Users collection indexes
            await cls.database.users.create_index("username", unique=True)
            await cls.database.users.create_index("email", unique=True)
            
            # Reports collection indexes
            await cls.database.reports.create_index([("user_id", 1), ("created_at", -1)])
            await cls.database.reports.create_index("status")
            await cls.database.reports.create_index("created_at")
            
            logger.info("✅ Database indexes created")
            
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")

    @classmethod
    def get_collection(cls, name: str):
        """Get collection by name"""
        return cls.database[name]

    @classmethod
    async def get_user_by_username(cls, username: str):
        """Get user by username"""
        collection = cls.get_collection("users")
        return await collection.find_one({"username": username})

    @classmethod
    async def get_user_by_email(cls, email: str):
        """Get user by email"""
        collection = cls.get_collection("users")
        return await collection.find_one({"email": email})

    @classmethod
    async def get_user_by_id(cls, user_id: str):
        """Get user by ID"""
        from bson import ObjectId
        collection = cls.get_collection("users")
        return await collection.find_one({"_id": ObjectId(user_id)})

# Database dependency
async def get_database():
    return MongoDB.database