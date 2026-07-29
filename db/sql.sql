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
)