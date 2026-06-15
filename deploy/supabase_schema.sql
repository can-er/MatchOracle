BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> 478127bb6eff

CREATE TABLE accuracy_snapshots (
    id UUID NOT NULL, 
    agent_name VARCHAR(64) NOT NULL, 
    accuracy FLOAT NOT NULL, 
    calibration FLOAT, 
    sample_size INTEGER NOT NULL, 
    weight_after FLOAT, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    tenant_id VARCHAR(64), 
    PRIMARY KEY (id)
);

CREATE INDEX ix_accuracy_agent_time ON accuracy_snapshots (agent_name, created_at);

CREATE INDEX ix_accuracy_snapshots_created_at ON accuracy_snapshots (created_at);

CREATE INDEX ix_accuracy_snapshots_tenant_id ON accuracy_snapshots (tenant_id);

CREATE TABLE audit_logs (
    id UUID NOT NULL, 
    actor VARCHAR(128), 
    action VARCHAR(128) NOT NULL, 
    resource VARCHAR(256), 
    detail JSONB, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    tenant_id VARCHAR(64), 
    PRIMARY KEY (id)
);

CREATE INDEX ix_audit_logs_created_at ON audit_logs (created_at);

CREATE INDEX ix_audit_logs_tenant_id ON audit_logs (tenant_id);

CREATE TABLE connectors (
    id UUID NOT NULL, 
    name VARCHAR(128) NOT NULL, 
    type VARCHAR(32) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    configuration JSONB NOT NULL, 
    last_checked_at TIMESTAMP WITHOUT TIME ZONE, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    tenant_id VARCHAR(64), 
    PRIMARY KEY (id), 
    CONSTRAINT uq_connector_name_tenant UNIQUE (name, tenant_id)
);

CREATE INDEX ix_connectors_name ON connectors (name);

CREATE INDEX ix_connectors_tenant_id ON connectors (tenant_id);

CREATE TABLE predictions (
    id UUID NOT NULL, 
    entity VARCHAR(512) NOT NULL, 
    domain VARCHAR(64), 
    prediction VARCHAR(64) NOT NULL, 
    score FLOAT NOT NULL, 
    confidence FLOAT NOT NULL, 
    risk_level VARCHAR(32), 
    explanation TEXT, 
    contributors JSONB, 
    weights_used JSONB, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    tenant_id VARCHAR(64), 
    PRIMARY KEY (id)
);

CREATE INDEX ix_predictions_created_at ON predictions (created_at);

CREATE INDEX ix_predictions_entity ON predictions (entity);

CREATE INDEX ix_predictions_tenant_id ON predictions (tenant_id);

CREATE TABLE tenants (
    id VARCHAR(64) NOT NULL, 
    name VARCHAR(128) NOT NULL, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id)
);

CREATE TABLE users (
    id UUID NOT NULL, 
    username VARCHAR(128) NOT NULL, 
    hashed_password VARCHAR(256) NOT NULL, 
    role VARCHAR(32) NOT NULL, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    tenant_id VARCHAR(64), 
    PRIMARY KEY (id), 
    CONSTRAINT uq_user_name_tenant UNIQUE (username, tenant_id)
);

CREATE INDEX ix_users_tenant_id ON users (tenant_id);

CREATE TABLE agent_results (
    id UUID NOT NULL, 
    prediction_id UUID NOT NULL, 
    agent_name VARCHAR(64) NOT NULL, 
    score FLOAT NOT NULL, 
    confidence FLOAT NOT NULL, 
    weight FLOAT, 
    reasoning TEXT, 
    extra JSONB, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    tenant_id VARCHAR(64), 
    PRIMARY KEY (id), 
    FOREIGN KEY(prediction_id) REFERENCES predictions (id) ON DELETE CASCADE
);

CREATE INDEX ix_agent_results_agent_name ON agent_results (agent_name);

CREATE INDEX ix_agent_results_prediction_id ON agent_results (prediction_id);

CREATE INDEX ix_agent_results_tenant_id ON agent_results (tenant_id);

CREATE TABLE feedback (
    id UUID NOT NULL, 
    prediction_id UUID NOT NULL, 
    validator VARCHAR(128), 
    verdict VARCHAR(32) NOT NULL, 
    corrected_prediction VARCHAR(64), 
    reward FLOAT, 
    comment TEXT, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    tenant_id VARCHAR(64), 
    PRIMARY KEY (id), 
    FOREIGN KEY(prediction_id) REFERENCES predictions (id) ON DELETE CASCADE
);

CREATE INDEX ix_feedback_prediction_id ON feedback (prediction_id);

CREATE INDEX ix_feedback_tenant_id ON feedback (tenant_id);

CREATE TABLE outcomes (
    id UUID NOT NULL, 
    prediction_id UUID NOT NULL, 
    actual VARCHAR(64) NOT NULL, 
    actual_score FLOAT, 
    correct BOOLEAN, 
    notes TEXT, 
    recorded_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    tenant_id VARCHAR(64), 
    PRIMARY KEY (id), 
    FOREIGN KEY(prediction_id) REFERENCES predictions (id) ON DELETE CASCADE, 
    UNIQUE (prediction_id)
);

CREATE INDEX ix_outcomes_tenant_id ON outcomes (tenant_id);

INSERT INTO alembic_version (version_num) VALUES ('478127bb6eff') RETURNING alembic_version.version_num;

-- Running upgrade 478127bb6eff -> dd35aa8af719

ALTER TABLE predictions ADD COLUMN score_detail JSONB;

UPDATE alembic_version SET version_num='dd35aa8af719' WHERE alembic_version.version_num = '478127bb6eff';

-- Running upgrade dd35aa8af719 -> c1d2e3f4a5b6

CREATE TABLE kv_store (
    key VARCHAR(255) NOT NULL, 
    value JSONB, 
    expires_at FLOAT, 
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (key)
);

UPDATE alembic_version SET version_num='c1d2e3f4a5b6' WHERE alembic_version.version_num = 'dd35aa8af719';

COMMIT;

