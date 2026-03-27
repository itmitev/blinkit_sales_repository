# Database Model

## Overview
This database contains sales transaction data from an online grocery delivery platform. It provides valuable insights into customer purchasing behavior, product demand, revenue trends, and sales performance over time.

---

## Tables

### customers
| Column 				 | Data Type 				| Nullable 	  | Default 	  |
|------------------------|--------------------------|-------------|---------------|
| customer_id            | bigint                   | NO          | —             |
| customer_name          | text                     | YES         | —             |
| email                  | text                     | YES         | —             |
| phone                  | bigint                   | YES         | —             |
| address                | text                     | YES         | —             |
| area                   | text                     | YES         | —             |
| zipcode                | bigint                   | YES         | —             |
| registration_date      | text                     | YES         | —             |
| customer_segment       | text                     | YES         | —             |
| total_orders           | bigint                   | YES         | —             |
| avg_order_value        | double precision         | YES         | —             |

---

### orders
| Column 				 | Data Type 				| Nullable 	  | Default 	  |
|------------------------|--------------------------|-------------|---------------|
| order_id               | bigint                   | NO          | —             |
| customer_id            | bigint                   | YES         | —             |
| order_date             | timestamp with time zone | YES         | —             |
| promised_delivery_time | timestamp with time zone | YES         | —             |
| actual_delivery_time   | timestamp with time zone | YES         | —             |
| delivery_status        | text                     | YES         | —             |
| order_total            | double precision         | YES         | —             |
| payment_method         | text                     | YES         | —             |
| delivery_partner_id    | bigint                   | YES         | —             |
| store_id               | bigint                   | YES         | —             |

---

### order_items
| Column 				 | Data Type 				| Nullable 	  | Default 	  |
|------------------------|--------------------------|-------------|---------------|
| order_id               | bigint                   | YES         | —             |
| product_id             | bigint                   | YES         | —             |
| quantity               | bigint                   | YES         | —             |
| unit_price             | double precision         | YES         | —             |

---

### products
| Column 				 | Data Type 				| Nullable 	  | Default 	  |
|------------------------|--------------------------|-------------|---------------|
| product_id             | bigint                   | NO          | —             |
| product_name           | text                     | YES         | —             |
| category               | text                     | YES         | —             |
| brand                  | text                     | YES         | —             |
| price                  | double precision         | YES         | —             |
| mrp                    | double precision         | YES         | —             |
| margin_percentage      | double precision         | YES         | —             |
| shelf_life_days        | bigint                   | YES         | —             |
| min_stock_level        | bigint                   | YES         | —             |
| max_stock_level        | bigint                   | YES         | —             |

---

### customer_feedback
| Column 				 | Data Type 				| Nullable 	  | Default 	  |
|------------------------|--------------------------|-------------|---------------|
| feedback_id            | bigint                   | NO          | —             |
| order_id               | bigint                   | YES         | —             |
| customer_id            | bigint                   | YES         | —             |
| rating                 | bigint                   | YES         | —             |
| feedback_text          | text                     | YES         | —             |
| feedback_category      | text                     | YES         | —             |
| sentiment              | text                     | YES         | —             |
| feedback_date          | text                     | YES         | —             |

---

### marketing_performance
| Column 				 | Data Type 				| Nullable 	  | Default 	  |
|------------------------|--------------------------|-------------|---------------|
| campaign_id            | bigint                   | NO          | —             |
| campaign_name          | text                     | YES         | —             |
| date                   | text                     | YES         | —             |
| target_audience        | text                     | YES         | —             |
| channel                | text                     | YES         | —             |
| impressions            | bigint                   | YES         | —             |
| clicks                 | bigint                   | YES         | —             |
| conversions            | bigint                   | YES         | —             |
| spend                  | double precision         | YES         | —             |
| revenue_generated      | double precision         | YES         | —             |
| roas                   | double precision         | YES         | —             |

---

## Relationships
- `orders.customer_id` → `customers.customer_id`
- `order_items.order_id` → `orders.order_id`
- `order_items.product_id` → `products.product_id`
- `customer_feedback.order_id` → `orders.order_id`

---

## Key Views
| View | Description |
|------|-------------|
| vw_monthly_sales | Monthly revenue aggregated by region |
| vw_customer_cohorts | Customer segmentation by registration month |

---

## Data Source
- **Source:** Supabase PostgreSQL (https://supabase.com/dashboard/project/khcnudaxsvauxbirokmo)
- **dbdocs.io:** https://dbdocs.io/itmitev/blinkit_sales_project?schema=public&view=relationships&table=customers
- **Project:** Blinkit Sales Project (Notion page link)
- **Last updated:** 2026-03-27

