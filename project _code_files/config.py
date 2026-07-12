# config.py - Configuration file for AI Resume Matcher

import os

class Config:
    """Main configuration class for the Resume Matcher application"""
    
    # Model configurations
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    
    # Scoring weights (must sum to 1.0)
    SEMANTIC_WEIGHT = 0.4
    TFIDF_WEIGHT = 0.3
    SKILLS_WEIGHT = 0.3
    
    # Match category thresholds
    EXCELLENT_THRESHOLD = 0.8
    GOOD_THRESHOLD = 0.6
    FAIR_THRESHOLD = 0.4
    
    # File paths
    MODEL_PATH = "resume_matcher_model.pkl"
    NLTK_DATA_PATH = os.path.expanduser("~/nltk_data")
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_RELOAD = True
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        'page_title': 'AI Resume Matcher',
        'page_icon': '📄',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # File upload limits
    MAX_FILE_SIZE_MB = 10
    MAX_BATCH_FILES = 10
    ALLOWED_FILE_TYPES = ['pdf', 'docx', 'txt']
    
    # Skills database - Comprehensive list of technical skills
    TECH_SKILLS = [
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
        'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
        'objective-c', 'dart', 'lua', 'haskell', 'clojure', 'f#', 'erlang',
        
        # Web Development - Frontend
        'html', 'css', 'sass', 'scss', 'less', 'react', 'angular', 'vue.js',
        'svelte', 'ember.js', 'backbone.js', 'jquery', 'bootstrap', 'tailwind',
        'material-ui', 'chakra-ui', 'ant-design', 'webpack', 'parcel', 'vite',
        
        # Web Development - Backend
        'node.js', 'express', 'fastapi', 'django', 'flask', 'spring', 'spring boot',
        'laravel', 'symfony', 'codeigniter', 'rails', 'sinatra', 'asp.net',
        'nest.js', 'koa.js', 'fastify', 'gin', 'fiber', 'echo',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'sqlite', 'mongodb', 'redis', 'elasticsearch',
        'oracle', 'sql server', 'cassandra', 'dynamodb', 'neo4j', 'influxdb',
        'couchdb', 'firebase', 'supabase', 'planetscale', 'cockroachdb',
        
        # Cloud Platforms
        'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'vercel', 'netlify',
        'digitalocean', 'linode', 'vultr', 'cloudflare', 'oracle cloud',
        
        # Cloud Services
        'ec2', 's3', 'lambda', 'rds', 'dynamodb', 'cloudfront', 'route53',
        'iam', 'vpc', 'ecs', 'eks', 'fargate', 'api gateway', 'cloudwatch',
        'azure functions', 'azure storage', 'azure sql', 'google app engine',
        'google cloud functions', 'google cloud storage', 'bigquery',
        
        # DevOps & Infrastructure
        'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions',
        'circleci', 'travis ci', 'terraform', 'ansible', 'puppet', 'chef',
        'vagrant', 'packer', 'consul', 'vault', 'prometheus', 'grafana',
        'elk stack', 'splunk', 'datadog', 'new relic', 'helm', 'istio',
        
        # Version Control
        'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
        
        # Operating Systems
        'linux', 'ubuntu', 'centos', 'redhat', 'debian', 'alpine', 'windows',
        'macos', 'unix', 'bash', 'powershell', 'zsh', 'fish',
        
        # Data Science & Machine Learning
        'machine learning', 'deep learning', 'artificial intelligence',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda', 'opencv',
        'nltk', 'spacy', 'transformers', 'hugging face', 'langchain',
        'computer vision', 'natural language processing', 'reinforcement learning',
        
        # Data Engineering
        'apache spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'snowflake',
        'redshift', 'bigquery', 'databricks', 'apache beam', 'flink',
        'storm', 'kinesis', 'pub/sub', 'event hubs',
        
        # Analytics & BI
        'tableau', 'power bi', 'looker', 'qlik', 'metabase', 'superset',
        'google analytics', 'mixpanel', 'amplitude', 'segment',
        
        # Mobile Development
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic',
        'cordova', 'phonegap', 'native script', 'expo',
        
        # Testing
        'unit testing', 'integration testing', 'e2e testing', 'selenium',
        'cypress', 'jest', 'mocha', 'chai', 'junit', 'pytest', 'rspec',
        'cucumber', 'postman', 'insomnia', 'jmeter', 'loadrunner',
        
        # Security
        'cybersecurity', 'penetration testing', 'owasp', 'ssl/tls',
        'oauth', 'jwt', 'saml', 'ldap', 'active directory', 'kerberos',
        'encryption', 'hashing', 'firewall', 'vpn', 'iam', 'zero trust',
        
        # API Technologies
        'rest api', 'graphql', 'grpc', 'soap', 'websockets', 'sse',
        'api gateway', 'swagger', 'openapi', 'postman', 'insomnia',
        
        # Message Queues
        'rabbitmq', 'apache kafka', 'redis pub/sub', 'amazon sqs',
        'google pub/sub', 'azure service bus', 'nats', 'pulsar',
        
        # Search Engines
        'elasticsearch', 'solr', 'algolia', 'amazon cloudsearch',
        'azure cognitive search', 'google cloud search',
        
        # Monitoring & Logging
        'prometheus', 'grafana', 'elk stack', 'splunk', 'datadog',
        'new relic', 'cloudwatch', 'azure monitor', 'google cloud monitoring',
        'sentry', 'rollbar', 'bugsnag', 'logz.io',
        
        # Blockchain & Web3
        'blockchain', 'ethereum', 'bitcoin', 'solidity', 'web3', 'smart contracts',
        'defi', 'nft', 'ipfs', 'polygon', 'binance smart chain',
        
        # Game Development
        'unity', 'unreal engine', 'godot', 'gamemaker', 'construct',
        'phaser', 'three.js', 'babylonjs', 'pygame', 'love2d',
        
        # Design & UI/UX
        'figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator',
        'after effects', 'principle', 'framer', 'invision', 'zeplin',
        
        # Project Management & Collaboration
        'jira', 'confluence', 'trello', 'asana', 'monday.com', 'notion',
        'slack', 'microsoft teams', 'discord', 'zoom', 'miro', 'figma',
        
        # Methodologies & Practices
        'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd',
        'microservices', 'serverless', 'event-driven architecture',
        'domain-driven design', 'clean architecture', 'solid principles',
        'design patterns', 'mvc', 'mvp', 'mvvm', 'flux', 'redux',
        
        # E-commerce
        'shopify', 'woocommerce', 'magento', 'prestashop', 'bigcommerce',
        'stripe', 'paypal', 'square', 'razorpay', 'payu',
        
        # CMS
        'wordpress', 'drupal', 'joomla', 'contentful', 'strapi',
        'ghost', 'jekyll', 'hugo', 'gatsby', 'next.js', 'nuxt.js',
        
        # Networking
        'tcp/ip', 'http/https', 'dns', 'load balancing', 'cdn',
        'nginx', 'apache', 'haproxy', 'cloudflare', 'akamai',
        
        # Protocols & Standards
        'json', 'xml', 'yaml', 'csv', 'protobuf', 'avro', 'parquet',
        'odata', 'jsonapi', 'hal', 'collection+json'
    ]
    
    # Soft skills and domain expertise
    SOFT_SKILLS = [
        'leadership', 'communication', 'teamwork', 'problem solving',
        'critical thinking', 'project management', 'time management',
        'adaptability', 'creativity', 'analytical thinking', 'mentoring',
        'collaboration', 'innovation', 'strategic thinking'
    ]
    
    # Industry domains
    DOMAINS = [
        'fintech', 'healthcare', 'e-commerce', 'education', 'gaming',
        'social media', 'enterprise software', 'cybersecurity', 'iot',
        'automotive', 'aerospace', 'telecommunications', 'media',
        'real estate', 'logistics', 'retail', 'manufacturing'
    ]
    
    @classmethod
    def get_all_skills(cls):
        """Get combined list of all skills"""
        return cls.TECH_SKILLS + cls.SOFT_SKILLS + cls.DOMAINS
    
    @classmethod
    def get_match_category(cls, score):
        """Get match category based on score"""
        if score >= cls.EXCELLENT_THRESHOLD:
            return "Excellent Match"
        elif score >= cls.GOOD_THRESHOLD:
            return "Good Match"
        elif score >= cls.FAIR_THRESHOLD:
            return "Fair Match"
        else:
            return "Poor Match"
    
    @classmethod
    def validate_weights(cls):
        """Validate that scoring weights sum to 1.0"""
        total = cls.SEMANTIC_WEIGHT + cls.TFIDF_WEIGHT + cls.SKILLS_WEIGHT
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")
        return True

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    API_RELOAD = True
    MODEL_PATH = "models/resume_matcher_model_dev.pkl"

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    API_RELOAD = False
    MODEL_PATH = "/app/models/resume_matcher_model.pkl"
    
    # Production-specific settings
    MAX_BATCH_FILES = 5  # Reduced for production
    API_WORKERS = 4

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    MODEL_PATH = "tests/test_model.pkl"
    MAX_BATCH_FILES = 2

# Configuration factory
def get_config(env='development'):
    """Get configuration based on environment"""
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    return configs.get(env, DevelopmentConfig)

# Validate configuration on import
if __name__ == "__main__":
    Config.validate_weights()
    print("✅ Configuration validation passed")
    print(f"📊 Total technical skills: {len(Config.TECH_SKILLS)}")
    print(f"🤝 Total soft skills: {len(Config.SOFT_SKILLS)}")
    print(f"🏢 Total domains: {len(Config.DOMAINS)}")
    print(f"🎯 Total skills database: {len(Config.get_all_skills())}")