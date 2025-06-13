import os
import shutil
import sys

def clear_run():
    """
    Deletes everything under the specified directories:
    - data/graph/transitions
    - data/graph/V_soft
    - data/graph/wind_averages
    - data/graph/trajectories
    """
    directories_to_clear = [
        "data/graph/transitions",
        "data/graph/V_soft", 
        "data/graph/wind_averages",
        "data/graph/trajectories"
    ]
    
    print("Starting cleanup process...")
    
    for directory in directories_to_clear:
        if os.path.exists(directory):
            try:
                # Remove all contents of the directory
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print(f"✓ Cleared contents of {directory}")
            except Exception as e:
                print(f"✗ Error clearing {directory}: {e}")
        else:
            print(f"⚠ Directory {directory} does not exist")
    
    print("Cleanup process completed!")

def show_menu():
    """Display the main menu options"""
    print("\n" + "="*50)
    print("           HOUSEKEEPING MENU")
    print("="*50)
    print("1. Clear Run Data")
    print("   (Deletes data/graph/transitions, V_soft, wind_averages, trajectories)")
    print("2. Exit")
    print("="*50)

def main():
    """Main program loop"""
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-2): ").strip()
            
            if choice == "1":
                print("\nYou selected: Clear Run Data")
                confirm = input("Are you sure you want to delete all run data? (y/N): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    clear_run()
                else:
                    print("Operation cancelled.")
                    
            elif choice == "2":
                print("\nExiting housekeeping script. Goodbye!")
                sys.exit(0)
                
            else:
                print("\n⚠ Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ An error occurred: {e}")
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
