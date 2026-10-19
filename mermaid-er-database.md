---
title: "CHAI-TEA Database Design"
config:
    layout: elk
    elk:
        nodePlacementStrategy: SIMPLE
---
erDiagram
    RESULT |o..o{ ANNOTATION : has_annotation
    ANNOTATION |o..o| ANNOTATION : has_annotation
    RESULT }o..o| USER : from_editor
    RESULT |o--o{ RESULT : from_previous
    WORKFLOW |o--o{ WORKFLOW : from_previous
    RESULT }|--|| WORKFLOW_RUN : of_run
    WORKFLOW_RUN }|--|| WORKFLOW : of_workflow
    WORKFLOW }|--|| PROJECT : of_project
    WORKFLOW ||--o{ WF_PERMISSION : has_permissions
    WF_PERMISSION }o--|| ROLE : allows_role
    WF_PERMISSION }o--|| USER : for_user
    PROJECT ||--o{ PJ_PERMISSION : has_permissions
    PJ_PERMISSION }o--|| ROLE : allows_role
    PJ_PERMISSION }o--|| USER : for_user

    ANNOTATION {
        uuid id PK
        uuid target_result_id FK
        uuid target_annotation_id FK
        uuid user_id FK
        string flag
        string comment
        JSON metadata
        JSON extra_data
        datetime md_timestamp
    }
    RESULT {
        uuid id PK
        uuid workflow_run_id FK
        string process_id
        string input
        string input_hash
        string input_segment
        int input_sequence
        bool was_successful
        JSON value
        JSON metadata
        JSON extra_data
        uuid previous_id FK
        uuid editor_user_id FK
        float md_cost
        float md_duration
        datetime md_timestamp
        datetime created
        datetime modified
    }
    WORKFLOW_RUN {
        uuid id PK
        uuid workflow_id FK
        bool was_successful
        JSON metadata
        JSON extra_data
        float duration
        datetime last_run
        datetime created
        datetime modified
    }
    WORKFLOW {
        uuid id PK
        uuid project_id FK
        uuid previous_id FK
        string name
        string description
        bool is_active
        datetime created
        datetime modified
    }
    PROJECT {
        uuid id PK
        string name
        string description
        bool is_active
        string auth_key
        datetime created
        datetime modified
        datetime planned_enddate
    }
    USER {
        uuid id PK
        string name
        string netid
        string email_address
        bool is_active
        string auth_key
        datetime created
        datetime modified
    }
    WF_PERMISSION {
        uuid id PK
        uuid user_id FK
        uuid workflow_id FK
        uuid role_id FK
        bool is_active
        datetime created
        datetime modified
    }
    PJ_PERMISSION {
        uuid id PK
        uuid user_id FK
        uuid project_id FK
        uuid role_id FK
        bool is_active
        datetime created
        datetime modified
    }
    ROLE {
        uuid id PK 
        string name
        string description
        bool is_active
        datetime created
        datetime modified
    }
