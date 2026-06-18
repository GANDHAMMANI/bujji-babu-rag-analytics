import csv

# Data
apps = [
    [1, "Expense Tracker", "Log and categorize expenses with spending charts"],
    [2, "Blog Platform", "CMS with auth, posts, comments, and tags"],
    [3, "Student Grade Manager", "Input marks, calculate grades, generate reports"],
    [4, "Job Application Tracker", "Track applications with status, notes, and deadlines"],
    [5, "Quiz App", "Timed MCQ quizzes with scoring and leaderboards"],
    [6, "Chat App", "Real-time messaging with WebSockets and rooms"],
    [7, "Recipe Finder", "Search recipes by ingredients using a food API"],
    [8, "Poll / Voting App", "Create polls, vote, and view live results with charts"],
    [9, "Habit Tracker", "Log daily habits with streaks and progress graphs"],
    [10, "Invoice Generator", "Create and download PDF invoices for freelancers"],
    [11, "Attendance Manager", "Mark and track student or employee attendance"],
    [12, "Movie Watchlist", "Search movies via TMDB API, save to watchlist with ratings"],
    [13, "Stock Portfolio Tracker", "Track stocks in real-time with profit/loss analytics"],
    [14, "Collaborative Whiteboard", "Real-time drawing and brainstorming board with multiple users"],
    [15, "Code Snippet Manager", "Save, tag, search, and share reusable code snippets"],
    [16, "Social Media Dashboard", "Unified dashboard to track posts, likes, and followers across platforms"],
    [17, "Online Exam Portal", "Full exam system with timer, auto-grading, and result reports"],
    [18, "Freelancer Project Manager", "Manage clients, projects, milestones, and invoices in one place"],
    [19, "Health & Fitness Tracker", "Log workouts, calories, water intake with progress charts"],
    [20, "Event Ticketing System", "Create events, sell tickets, generate QR codes for entry"],
    [21, "Multi-User Kanban Board", "Drag-and-drop task board with teams, roles, and deadlines"],

    [22, "Subscription Billing App", "Manage plans, payments, and renewals with Stripe integration"],
    [23, "Multi-Vendor Marketplace", "Multiple sellers list products with order tracking"],
    [24, "Real-Time Auction System", "Live bidding with countdown timers and winner alerts"],
    [25, "Document Collaboration Tool", "Real-time multi-user editing with version history"],
    [26, "Learning Management System", "Courses, videos, quizzes, and progress tracking"],
    [27, "Crowdfunding Platform", "Create campaigns with donation tracking and goal progress"],

    # Newly added now
    [28, "Blood Donation Management", "Register donors, track blood groups, manage requests and availability"],
    [29, "Hotel Room Booking System", "Browse rooms, make reservations, manage check-in and check-out"],
    [30, "Complaint Management System", "Submit, track, and resolve complaints with status updates and admin panel"]
]

# Write to CSV
with open("app_ideas.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    
    # Header
    writer.writerow(["ID", "App Name", "Description"])
    
    # Data rows
    writer.writerows(apps)

print("CSV file 'app_ideas.csv' created successfully!")