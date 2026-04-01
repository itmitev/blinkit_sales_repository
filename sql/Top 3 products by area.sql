with
  area_prod as (
    select
      c.area,
      p.product_name,
      sum(
        coalesce(oi.quantity, 0) * coalesce(p.price, 0)
      ) as revenue
    from
      customers c
      join orders o on o.customer_id = c.customer_id
      left join order_items oi on oi.order_id = o.order_id
      left join products p on p.product_id = oi.product_id
    group by
      1,
      2
  ),
  area_prod_ranked as (
    select
      ap.*,
      RANK() over (
        partition by
          area
        order by
          revenue desc
      ) revenue_rank
    from
      area_prod ap
  )
select
  *
from
  area_prod_ranked
  where revenue_rank <=3
