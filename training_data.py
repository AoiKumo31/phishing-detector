import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_text

def get_training_data():
    """
    Generate training data for the phishing email detector.
    This includes both phishing and legitimate emails.
    
    Returns:
        X_train: Training text data
        X_test: Testing text data
        y_train: Training labels
        y_test: Testing labels
    """
    # Generate legitimate email examples
    legitimate_emails = [
        "Hello John, I hope this email finds you well. I wanted to follow up on our meeting last week regarding the Q3 marketing plan. Attached is the revised budget proposal based on our discussion. Please review it and let me know your thoughts. Best regards, Sarah",
        "Dear Team, Please find attached the minutes from yesterday's weekly meeting. The next meeting is scheduled for Monday at 10:00 AM. Please review the action items assigned to you and provide updates by Friday. Thanks, Management",
        "Hi David, Thanks for your email. I'm available for a call tomorrow between 2-4 PM. Let me know what time works best for you. Regards, Jennifer",
        "Monthly Newsletter: Company Updates - New product launch next month - Employee of the month: Jane Smith - Upcoming holiday schedule - Reminder about annual review process",
        "Dear Customer, Thank you for your recent purchase. Your order #12345 has been shipped and is expected to arrive within 3-5 business days. Your tracking number is ABC123XYZ. If you have any questions, please contact our customer service. Thank you for shopping with us!",
        "Hi team, The client meeting scheduled for tomorrow has been postponed to next week. I'll send a calendar update shortly with the new date and time. Sorry for any inconvenience. Regards, Project Manager",
        "Reminder: The office will be closed on Monday for the national holiday. Regular operations will resume on Tuesday. Have a great long weekend!",
        "Good afternoon, I've reviewed the proposal you sent over and have a few comments. Can we schedule a quick call tomorrow to discuss? Let me know your availability. Thanks, Director",
        "IT Department Notice: We will be performing system maintenance this Saturday from 10 PM to 2 AM. During this time, the email server may be temporarily unavailable. We apologize for any inconvenience this may cause.",
        "Dear all, We're pleased to announce that the company picnic will be held on July 15th at Sunshine Park. Please RSVP by July 1st. Family members are welcome. HR Department",
        "Hello, This is a reminder that your subscription will renew automatically on 06/15/2023. If you would like to make any changes to your plan, please log in to your account at our official website or contact customer support at support@legitimatecompany.com. Thank you for your continued business.",
        "Dear valued customer, Your account statement for May 2023 is now available in your online banking portal. To view your statement, please log in to your account through our official website. For security reasons, we do not include links in these notification emails. If you have any questions, please contact our customer service department.",
        "Hi team, I've uploaded the project files to our secure company SharePoint. You can access them using your regular company credentials. Let me know if you have any trouble accessing the documents. Best, Project Coordinator",
        "Meeting Invitation: Quarterly Business Review Date: June 10, 2023 Time: 1:00 PM - 3:00 PM Location: Conference Room A Agenda: 1. Q2 Performance Review 2. Q3 Objectives 3. Open Discussion Please bring your department reports. Refreshments will be provided.",
        "Dear Dr. Williams, I'm writing to confirm your speaking engagement at our conference on September 5th. Please review the attached agenda and let me know if the scheduled time works for you. We'll need your presentation slides by August 20th. Thank you, Conference Organizer"
    ]
    
    # Generate phishing email examples
    phishing_emails = [
        "URGENT: Your account has been compromised! Click here to verify your information immediately: http://fake-bank-secure.com. Failure to respond within 24 hours will result in account termination.",
        "Dear Customer, We've detected suspicious activity on your account. Please verify your identity by providing your full name, account number, and social security number by replying to this email.",
        "Congratulations! You've won a $1,000 Amazon gift card! To claim your prize, click here: www.amazon-gifts.scam.com and enter your credit card details for verification purposes only.",
        "Your PayPal account has been limited! We need you to verify your information now to avoid account suspension. Click: http://paypa1.secure-verification.com",
        "ATTENTION: Your tax refund of $2,458.00 is ready for processing. Submit your banking details at: www.irs-refund-secure.com to receive your money within 24 hours.",
        "Apple Security Alert: Your Apple ID was used to sign in to iCloud on a new device. If this wasn't you, recover your account here: http://apple-secure-login.com",
        "Dear valued customer, Your Bank of America account will be suspended. We need you to update your account information. Please click this secure link: http://bank0famerica-secure.com",
        "Netflix: Your payment method has expired. To avoid service interruption, update your billing information here: www.netflix-accounts-billing.com",
        "IT Department: Your email storage is full. Click here to expand your mailbox storage capacity immediately: http://mail-storage-upgrade.com",
        "FINAL WARNING: Your internet service will be disconnected in 24 hours due to missed payment. To maintain service, update your payment information: www.internet-service-billing-secure.com",
        "Important HR Update: All employees must review and acknowledge the updated company policy by following this link: http://company-policies.secure-server.co.tk",
        "Package Delivery Notification: We attempted to deliver your package but were unable to. Schedule redelivery here: www.delivery-schedule-service.info/reschedule",
        "Your Office 365 password is about to expire. Reset it now by clicking: http://microsoft-365-password-reset.com to ensure uninterrupted access to your email.",
        "URGENT: Unauthorized purchase of $750 has been made with your credit card at Walmart. If this wasn't you, dispute this charge immediately: www.card-security-verification.com",
        "ATTENTION SHOPPERS: You have been selected to participate in our survey. Complete it now to receive a $100 Visa gift card: http://survey-rewards.info/walmart"
    ]
    
    # Create labels (0 for legitimate, 1 for phishing)
    legitimate_labels = [0] * len(legitimate_emails)
    phishing_labels = [1] * len(phishing_emails)
    
    # Combine the datasets
    all_emails = legitimate_emails + phishing_emails
    all_labels = legitimate_labels + phishing_labels
    
    # Preprocess all emails
    preprocessed_emails = [preprocess_text(email) for email in all_emails]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_emails,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels  # Maintain the same distribution of labels in both sets
    )
    
    return X_train, X_test, y_train, y_test
