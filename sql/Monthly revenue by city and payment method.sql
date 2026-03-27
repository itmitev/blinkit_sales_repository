select
  date_trunc('month', o.order_date) as mnth,
  c.area,
  c.zipcode,
  o.payment_method,
  sum(o.order_total) as revenue
from
  customers c
join orders o on o.customer_id = c.customer_id
group by
  1,
  2,
  3,
  4