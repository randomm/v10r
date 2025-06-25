-- v10r Mock Data for Integration Testing
-- This script creates test tables and inserts sample data

-- Test table 1: Articles (simulates content management)
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    author VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Vector columns (will be added by v10r)
    title_vector vector(768),
    title_embedding_model VARCHAR(255),
    title_vector_created_at TIMESTAMP,
    
    content_vector vector(768),
    content_embedding_model VARCHAR(255),
    content_vector_created_at TIMESTAMP,
    
    summary_vector vector(768),
    summary_embedding_model VARCHAR(255),
    summary_vector_created_at TIMESTAMP
);

-- Test table 2: Products (simulates e-commerce)
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    specifications TEXT,
    category VARCHAR(100),
    price DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Vector columns for multiple text fields
    name_vector vector(1024),
    name_embedding_model VARCHAR(255),
    name_vector_created_at TIMESTAMP,
    
    description_vector vector(1024),
    description_embedding_model VARCHAR(255),
    description_vector_created_at TIMESTAMP
);

-- Test table 3: User profiles (simulates social/job platform)
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    bio TEXT,
    skills TEXT,
    experience TEXT,
    location VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Single vector column for bio
    bio_vector vector(768),
    bio_embedding_model VARCHAR(255),
    bio_vector_created_at TIMESTAMP
);

-- Test table 4: Documents with HTML content (tests preprocessing)
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255),
    content_html TEXT,
    content_cleaned TEXT,  -- For preprocessing tests
    file_type VARCHAR(50),
    uploaded_at TIMESTAMP DEFAULT NOW(),
    
    -- Vector columns for cleaned content
    content_vector vector(768),
    content_embedding_model VARCHAR(255),
    content_vector_created_at TIMESTAMP,
    
    -- Preprocessing metadata
    content_cleaning_config VARCHAR(100),
    content_cleaned_at TIMESTAMP
);

-- Create indexes for test queries
CREATE INDEX idx_articles_author ON articles(author);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_documents_type ON documents(file_type);

-- Insert sample articles
INSERT INTO articles (title, content, summary, author) VALUES
    ('Introduction to Vector Databases', 
     'Vector databases are specialized database systems designed to store and query high-dimensional vectors efficiently. They are essential for modern AI applications including semantic search, recommendation systems, and similarity matching.',
     'Overview of vector databases and their applications in AI',
     'Jane Smith'),
     
    ('PostgreSQL pgvector Extension Guide',
     'The pgvector extension transforms PostgreSQL into a powerful vector database. It provides vector data types, indexing methods like IVFFlat and HNSW, and distance operators for similarity search.',
     'Complete guide to using pgvector with PostgreSQL',
     'John Doe'),
     
    ('Building AI-Powered Search Systems',
     'Modern search systems leverage embeddings to understand semantic meaning beyond keyword matching. This enables more intuitive and accurate search results.',
     'How to implement semantic search using embeddings',
     'Alice Johnson'),
     
    ('Docker Compose for Development Environments',
     'Docker Compose simplifies the orchestration of multi-container applications. It allows developers to define entire development stacks in a single YAML file.',
     'Using Docker Compose for consistent dev environments',
     'Bob Wilson'),
     
    ('Machine Learning in Production',
     'Deploying ML models in production requires careful consideration of scalability, monitoring, and reliability. This article covers best practices and common pitfalls.',
     'Best practices for production ML deployments',
     'Carol Davis');

-- Insert sample products
INSERT INTO products (name, description, specifications, category, price) VALUES
    ('Wireless Bluetooth Headphones',
     'Premium noise-cancelling wireless headphones with 30-hour battery life and superior sound quality.',
     'Bluetooth 5.0, Active Noise Cancellation, 30-hour battery, Quick charge, Foldable design',
     'Electronics',
     199.99),
     
    ('Ergonomic Office Chair',
     'Professional office chair with lumbar support, adjustable height, and breathable mesh backing for all-day comfort.',
     'Adjustable height: 17-21 inches, Weight capacity: 300lbs, Lumbar support, Breathable mesh, 5-year warranty',
     'Furniture',
     299.50),
     
    ('Smart Fitness Tracker',
     'Advanced fitness tracker with heart rate monitoring, GPS, and 7-day battery life. Track your workouts and health metrics.',
     'Heart rate monitor, GPS tracking, 7-day battery, Water resistant IPX7, Sleep tracking',
     'Wearables',
     149.99),
     
    ('Stainless Steel Water Bottle',
     'Insulated stainless steel water bottle that keeps drinks cold for 24 hours or hot for 12 hours.',
     'Double-wall vacuum insulation, 18/8 stainless steel, 32oz capacity, Leak-proof lid, BPA-free',
     'Lifestyle',
     34.95),
     
    ('Mechanical Gaming Keyboard',
     'RGB backlit mechanical keyboard with tactile switches, programmable keys, and gaming-focused features.',
     'Cherry MX Blue switches, RGB per-key lighting, N-key rollover, Programmable macros, USB-C connection',
     'Gaming',
     159.99);

-- Insert sample user profiles
INSERT INTO user_profiles (username, bio, skills, experience, location) VALUES
    ('tech_enthusiast',
     'Passionate software developer with 5 years of experience in full-stack development. Love working with modern technologies and solving complex problems.',
     'Python, JavaScript, React, PostgreSQL, Docker, AWS',
     'Senior Software Engineer at TechCorp, Previously Full-Stack Developer at StartupXYZ',
     'San Francisco, CA'),
     
    ('data_scientist_pro',
     'Data scientist specializing in machine learning and AI. Experienced in building predictive models and deploying ML solutions at scale.',
     'Python, R, TensorFlow, PyTorch, SQL, Spark, MLOps',
     'Lead Data Scientist at AI Innovations, PhD in Computer Science',
     'New York, NY'),
     
    ('ux_designer_creative',
     'Creative UX designer focused on user-centered design principles. Passionate about creating intuitive and accessible digital experiences.',
     'Figma, Adobe Creative Suite, User Research, Prototyping, Design Systems',
     '5+ years in UX Design at various agencies and startups',
     'Austin, TX'),
     
    ('devops_engineer',
     'DevOps engineer with expertise in cloud infrastructure, automation, and CI/CD pipelines. Helping teams ship faster and more reliably.',
     'Kubernetes, Docker, Terraform, Jenkins, AWS, monitoring, automation',
     'Senior DevOps Engineer, 7 years experience in infrastructure',
     'Seattle, WA');

-- Insert sample documents with HTML content (for preprocessing tests)
INSERT INTO documents (filename, content_html, file_type) VALUES
    ('introduction.html',
     '<html><head><title>Introduction</title></head><body><h1>Welcome to v10r</h1><p>This is a <strong>comprehensive</strong> guide to vector databases.</p><script>console.log("test");</script><p>Learn about <a href="/pgvector">pgvector</a> and more.</p></body></html>',
     'html'),
     
    ('guide.html',
     '<div class="content"><h2>Setup Guide</h2><p>Follow these steps:</p><ol><li>Install Docker</li><li>Run PostgreSQL</li><li>Enable pgvector</li></ol><footer>© 2024 v10r Project</footer></div>',
     'html'),
     
    ('readme.txt',
     'v10r: Generic PostgreSQL Notify-Based Vectorizer Service

This service provides automatic vectorization of text columns in PostgreSQL databases using any OpenAI-compatible embedding API.',
     'text'),
     
    ('complex.html',
     '<html><head><meta charset="utf-8"><title>Complex Document</title><style>.hidden{display:none}</style></head><body><nav><ul><li>Home</li><li>About</li></ul></nav><main><article><h1>Complex HTML Document</h1><p>This document contains various HTML elements including <em>emphasis</em>, <code>code snippets</code>, and <mark>highlighted text</mark>.</p><blockquote>This is a quote from an expert in the field.</blockquote><table><tr><th>Feature</th><th>Support</th></tr><tr><td>Vectors</td><td>Yes</td></tr></table></article></main><script src="analytics.js"></script></body></html>',
     'html');

-- Create test triggers for some tables (to verify trigger system)
-- These would normally be created by the v10r service

-- Trigger for articles table (title column)
CREATE TRIGGER articles_title_vector_trigger
    AFTER INSERT OR UPDATE ON articles
    FOR EACH ROW
    WHEN (
        NEW.title_vector IS NULL OR
        (TG_OP = 'UPDATE' AND OLD.title IS DISTINCT FROM NEW.title)
    )
    EXECUTE FUNCTION v10r.generic_vector_notify('v10r_events', 'articles_title', 'title_embedding_model', 'title');

-- Trigger for products table (name column)
CREATE TRIGGER products_name_vector_trigger
    AFTER INSERT OR UPDATE ON products
    FOR EACH ROW
    WHEN (
        NEW.name_vector IS NULL OR
        (TG_OP = 'UPDATE' AND OLD.name IS DISTINCT FROM NEW.name)
    )
    EXECUTE FUNCTION v10r.generic_vector_notify('v10r_events', 'products_name', 'name_embedding_model', 'name');

-- Insert test cases for NULL vector scenarios
INSERT INTO articles (title, content, summary, author) VALUES
    ('Test Article with NULL Vector', 'This article should trigger vectorization', 'Test summary', 'Test Author');

INSERT INTO products (name, description, category, price) VALUES
    ('Test Product', 'This product should also trigger vectorization', 'Test Category', 99.99);

-- Create test configuration entries
INSERT INTO v10r_metadata.column_registry (
    database_name, schema_name, table_name, 
    original_column_name, actual_column_name, 
    column_type, dimension, config_key
) VALUES
    ('v10r_test_db', 'public', 'articles', 'title_vector', 'title_vector', 'vector', 768, 'infinity_static'),
    ('v10r_test_db', 'public', 'articles', 'content_vector', 'content_vector', 'vector', 768, 'infinity_static'),
    ('v10r_test_db', 'public', 'products', 'name_vector', 'name_vector', 'vector', 1024, 'infinity_bge'),
    ('v10r_test_db', 'public', 'products', 'description_vector', 'description_vector', 'vector', 1024, 'infinity_bge'),
    ('v10r_test_db', 'public', 'user_profiles', 'bio_vector', 'bio_vector', 'vector', 768, 'openai_small');

-- Log mock data completion
INSERT INTO v10r_metadata.vectorization_log (
    database_name, schema_name, table_name, status, created_at, completed_at
) VALUES (
    current_database(), 'public', 'mock_data_complete', 'success', NOW(), NOW()
); 