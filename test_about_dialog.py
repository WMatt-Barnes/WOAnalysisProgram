#!/usr/bin/env python3
"""
Test script to verify the About dialog icon display.
"""

import tkinter as tk
from tkinter import ttk
import os
import sys

def test_about_dialog():
    """Test the About dialog icon display"""
    root = tk.Tk()
    root.title("About Dialog Test")
    root.geometry("400x200")
    
    # Set the main window icon
    try:
        icon_path = os.path.join(os.path.dirname(__file__), 'icons', 'app_icon.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
            print(f"✓ Main window icon loaded from: {icon_path}")
        else:
            print(f"✗ Icon file not found at: {icon_path}")
    except Exception as e:
        print(f"✗ Could not load main window icon: {e}")
    
    def show_about():
        """Show about dialog with icon"""
        # Create custom about window
        about_window = tk.Toplevel(root)
        about_window.title("About Work Order Analysis Pro")
        about_window.geometry("600x500")
        about_window.transient(root)
        about_window.grab_set()
        about_window.resizable(False, False)
        
        # Center the window
        about_window.update_idletasks()
        x = (about_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (about_window.winfo_screenheight() // 2) - (500 // 2)
        about_window.geometry(f"600x500+{x}+{y}")
        
        # Set icon for about window
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'icons', 'app_icon.ico')
            if os.path.exists(icon_path):
                about_window.iconbitmap(icon_path)
                print("✓ About window icon set")
        except Exception as e:
            print(f"✗ Could not load icon for about window: {e}")
        
        # Main frame
        main_frame = ttk.Frame(about_window, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Icon and title frame
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Create a dedicated frame for the icon
        icon_frame = ttk.Frame(header_frame)
        icon_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        # Try to display the icon with better error handling
        icon_displayed = False
        try:
            # Try the 64x64 version first
            icon_path = os.path.join(os.path.dirname(__file__), 'icons', 'app_icon_64x64.png')
            if os.path.exists(icon_path):
                from PIL import Image, ImageTk
                icon_img = Image.open(icon_path)
                # Ensure the image is the right size
                icon_img = icon_img.resize((64, 64), Image.Resampling.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon_img)
                
                # Create a label with a border to make it more visible
                icon_label = tk.Label(icon_frame, image=icon_photo, relief=tk.RAISED, bd=2)
                icon_label.image = icon_photo  # Keep a reference
                icon_label.pack()
                icon_displayed = True
                print("✓ Icon displayed successfully in About dialog")
            else:
                print(f"✗ Icon file not found at: {icon_path}")
        except Exception as e:
            print(f"✗ Could not display icon in about dialog: {e}")
            # Try alternative icon size
            try:
                icon_path = os.path.join(os.path.dirname(__file__), 'icons', 'app_icon_128x128.png')
                if os.path.exists(icon_path):
                    from PIL import Image, ImageTk
                    icon_img = Image.open(icon_path)
                    icon_img = icon_img.resize((64, 64), Image.Resampling.LANCZOS)
                    icon_photo = ImageTk.PhotoImage(icon_img)
                    
                    icon_label = tk.Label(icon_frame, image=icon_photo, relief=tk.RAISED, bd=2)
                    icon_label.image = icon_photo
                    icon_label.pack()
                    icon_displayed = True
                    print("✓ Alternative icon displayed successfully")
            except Exception as e2:
                print(f"✗ Could not display alternative icon: {e2}")
        
        # If no icon was displayed, show a placeholder
        if not icon_displayed:
            placeholder_label = tk.Label(icon_frame, text="📊", font=('Arial', 48), 
                                       relief=tk.RAISED, bd=2, bg='lightgray')
            placeholder_label.pack()
            print("⚠ Using placeholder icon")
        
        # Title and version frame
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        # Application title
        title_label = ttk.Label(title_frame, text="Work Order Analysis Pro", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Version
        version_label = ttk.Label(title_frame, text="Version 2.0", 
                                 font=('Arial', 14))
        version_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Subtitle
        subtitle_label = ttk.Label(title_frame, text="AI-Powered Failure Mode Classification System", 
                                  font=('Arial', 11, 'italic'), foreground='gray')
        subtitle_label.pack(anchor=tk.W)
        
        # Separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=(0, 20))
        
        # About text in a scrollable frame
        text_frame = ttk.LabelFrame(main_frame, text="Features", padding=15)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create scrollable text widget
        text_widget = tk.Text(text_frame, wrap=tk.WORD, height=12, 
                             font=('Arial', 10), relief=tk.FLAT, 
                             background=text_frame.cget('background'),
                             padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # About text content
        about_text = """• Intelligent failure code assignment using AI
• SpaCy NLP for advanced linguistic analysis  
• Sentence embeddings for semantic similarity
• Crow-AMSAA reliability analysis
• Risk assessment and calculation
• Comprehensive reporting and export
• Weibull analysis and PM optimization
• Spares analysis and recommendations
• Advanced filtering and data management
• Multi-format export capabilities

Developed by Matt Barnes
© 2025 All Rights Reserved"""
        
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Close button
        close_button = ttk.Button(button_frame, text="Close", command=about_window.destroy)
        close_button.pack(side=tk.RIGHT)
        
        # Focus on the close button
        close_button.focus_set()
        
        # Bind Enter key to close
        about_window.bind('<Return>', lambda e: about_window.destroy())
        about_window.bind('<Escape>', lambda e: about_window.destroy())
    
    # Main test interface
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(main_frame, text="About Dialog Icon Test", font=('Arial', 16, 'bold')).pack(pady=(0, 20))
    
    ttk.Label(main_frame, text="Click the button below to test the About dialog with icon display:").pack(pady=(0, 20))
    
    ttk.Button(main_frame, text="Show About Dialog", command=show_about).pack(pady=(0, 20))
    
    ttk.Label(main_frame, text="Check the console for detailed status messages.", font=('Arial', 9)).pack()
    
    print("About dialog test started. Click 'Show About Dialog' to test the icon display.")
    root.mainloop()

if __name__ == "__main__":
    test_about_dialog() 