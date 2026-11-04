create table video_audit_task(
    id int primary key auto_increment,
    task_status tinyint unsigned default 0 not null comment '任务状态：0-已接收 1-进行中 2-已完成 3-异常',
    audit_type varchar(255) not null comment '审核类型',
    oss_url varchar(255) not null comment 'oss地址',
    create_time  timestamp default CURRENT_TIMESTAMP null comment '任务创建时间  服务器时间',
    update_time  timestamp default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP
);

create table video_audit_task_result(
    id int primary key auto_increment,
    task_id int not null comment '关联 video_audit_task.id',
    score decimal(5,2) null comment '审核分数',
    result_json json null comment '审核结果原始JSON',
    create_time  timestamp default CURRENT_TIMESTAMP null comment '结果创建时间  服务器时间',
    update_time  timestamp default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    key idx_task_id (task_id)
);

create table score_rule_set(
    id int primary key auto_increment,
    audit_type varchar(255) not null comment '审核类型 对应 AuditType 的值',
    version int not null comment '同一审核类型的版本号 从1递增',
    kind varchar(16) not null default 'rubric' comment 'rubric-结构化评分项 text-整段文本规则',
    status varchar(16) not null default 'draft' comment 'draft-草稿 active-生效 archived-归档',
    source varchar(32) null comment '来源 oss-xlsx/oss-docx/local-txt/web',
    content text null comment 'kind=text 时的规则全文',
    remark varchar(255) null comment '变更说明',
    operator varchar(64) null comment '最后修改人',
    create_time timestamp default CURRENT_TIMESTAMP null comment '创建时间',
    update_time timestamp default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP comment '更新时间',
    published_at timestamp null comment '发布时间',
    unique key uk_rule_audit_version (audit_type, version),
    key idx_rule_type_status (audit_type, status)
);

create table score_rule_item(
    id int primary key auto_increment,
    rule_set_id int not null comment '关联 score_rule_set.id',
    dimension varchar(64) null comment '一级维度',
    number varchar(16) not null comment '评分项编号',
    point varchar(255) not null comment '细分评分点',
    evidence varchar(500) null comment '视频识别依据',
    remark varchar(255) null comment '备注 写入打分提示词作为该项补充判定说明',
    score decimal(6,2) not null comment '分值',
    sort_order int default 0 comment '展示与提示词中的顺序',
    key idx_rule_set_id (rule_set_id)
);