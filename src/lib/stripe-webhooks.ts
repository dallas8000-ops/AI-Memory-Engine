/** Stripe webhook event dispatcher — extend for your business logic */
export async function dispatchStripeEvent(event: { type: string; data: { object: unknown } }) {
  console.log(`[stripe] ${event.type}`);
  // Handle checkout.session.completed, customer.subscription.*, etc.
}
