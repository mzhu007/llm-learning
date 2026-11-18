import os
from agents.deals import Opportunity
from agents.agent import Agent
from litellm import completion
import requests

pushover_url = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE
    # MODEL = "anthropic/claude-haiku-5-5"
    MODEL = "openai/gpt-5-nano"

    def __init__(self):
        """
        Set up this object to either do push notifications via Pushover,
        or SMS via Twilio,
        whichever is specified in the constants
        """
        self.log("Messaging Agent is initializing")
        self.pushover_user = os.getenv("PUSHOVER_USER", "your-pushover-user-if-not-using-env")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "your-pushover-user-if-not-using-env")
        self.log("Messaging Agent has initialized Pushover and Claude")

    def push(self, text):
        """
        Send a Push Notification using the Pushover API
        """
        self.log("Messaging Agent is sending a push notification")
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": text,
            "sound": "cashregister",
        }
        requests.post(pushover_url, data=payload)

    # def notify_user_of_deal(description: str, deal_price: float, estimated_true_value: float, url: str) -> str:
    #     """
    #     This tool notifies the user of a great deal, given a description of it, the price of the deal, and the estimated true value
    #     """
    #     print(f"Fake function to notify user of {description} which costs {deal_price} and estimate is {estimated_true_value}")
    #     return "notification sent ok"

    def alert(self, opportunity: Opportunity):
        """
        This tool notifies the user of a great deal, given a description of it, the price of the deal, and the estimated true value
        """
        description = opportunity.deal.product_description[:10] + "... "
        deal_price = opportunity.deal.price
        estimated_true_value = opportunity.estimate
        print(f"Fake function to notify user of {description} which costs {deal_price} and estimate is {estimated_true_value}")
        return "notification sent ok"

    # def alert(self, opportunity: Opportunity):
    #     """
    #     Make an alert about the specified Opportunity
    #     """
    #     text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
    #     text += f"Estimate=${opportunity.estimate:.2f}, "
    #     text += f"Discount=${opportunity.discount:.2f} :"
    #     text += opportunity.deal.product_description[:10] + "... "
    #     text += opportunity.deal.url
    #     self.push(text)
    #     self.log("Messaging Agent has completed")

    def craft_message(
        self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        user_prompt = "Please summarize this great deal in 2-3 sentences to be sent as an exciting push notification alerting the user about this deal.\n"
        user_prompt += f"Item Description: {description}\nOffered Price: {deal_price}\nEstimated true value: {estimated_true_value}"
        user_prompt += "\n\nRespond only with the 2-3 sentence message which will be used to alert & excite the user about this deal"
        response = completion(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str):
        """
        Make an alert about the specified details
        """
        self.log("Messaging Agent is using Claude to craft the message")
        text = self.craft_message(description, deal_price, estimated_true_value)
        self.push(text[:200] + "... " + url)
        self.log("Messaging Agent has completed")
