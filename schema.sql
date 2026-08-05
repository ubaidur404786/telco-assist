-- Schema for HexaMobile text-to-SQL generation prompt
-- Source DB: telco.db

CREATE TABLE "plans" (
"plan_id" INTEGER,
  "plan_name" TEXT,
  "monthly_price" REAL,
  "data_gb" REAL,
  "voice_minutes" REAL,
  "sms_included" REAL,
  "is_unlimited_data" INTEGER,
  "contract_type" TEXT
);

CREATE TABLE "customers" (
"customer_id" INTEGER,
  "first_name" TEXT,
  "last_name" TEXT,
  "age" INTEGER,
  "city" TEXT,
  "region" TEXT,
  "plan_id" INTEGER,
  "signup_date" TEXT,
  "status" TEXT,
  "churn_date" TEXT
);

CREATE TABLE "usage" (
"usage_id" INTEGER,
  "customer_id" INTEGER,
  "month" TEXT,
  "data_used_gb" REAL,
  "voice_minutes_used" INTEGER,
  "sms_sent" INTEGER,
  "roaming_used" INTEGER
);

CREATE TABLE "bills" (
"bill_id" INTEGER,
  "customer_id" INTEGER,
  "month" TEXT,
  "amount" REAL,
  "status" TEXT
);

CREATE TABLE "tickets" (
"ticket_id" INTEGER,
  "customer_id" INTEGER,
  "created_date" TEXT,
  "category" TEXT,
  "channel" TEXT,
  "status" TEXT,
  "resolution_hours" REAL
);

