create view vw_monthly_sales as
select
  date_trunc('month', o.order_date) as mnth,
  c.area,
  c.zipcode,
  sum(coalesce(oi.quantity, 0) * coalesce(p.price, 0)) as revenue
from
  customers c
join orders o on o.customer_id = c.customer_id
left join order_items oi on oi.order_id = o.order_id
left join products p on p.product_id = oi.product_id
group by
  1,
  2,
  3