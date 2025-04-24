import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

def convert_text_to_pdf(input_file, output_file):
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create a PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles for different heading levels
    title_style = ParagraphStyle(
        name='CustomTitle', 
        parent=styles['Heading1'], 
        fontSize=18,
        alignment=TA_CENTER
    )
    
    heading2_style = ParagraphStyle(
        name='CustomHeading2', 
        parent=styles['Heading2'], 
        fontSize=16
    )
    
    heading3_style = ParagraphStyle(
        name='CustomHeading3', 
        parent=styles['Heading3'], 
        fontSize=14
    )
    
    normal_style = ParagraphStyle(
        name='CustomNormal', 
        parent=styles['Normal'], 
        fontSize=12,
        alignment=TA_LEFT
    )
    
    # Process the content
    lines = content.split('\n')
    story = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Handle headers
        if line.startswith('# '):
            story.append(Paragraph(line[2:], title_style))
            story.append(Spacer(1, 12))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], heading2_style))
            story.append(Spacer(1, 10))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], heading3_style))
            story.append(Spacer(1, 8))
        else:
            # Collect paragraph text (may span multiple lines)
            paragraph_text = line
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith('#'):
                paragraph_text += ' ' + lines[j].strip()
                j += 1
            i = j - 1  # Update i to the last line of the paragraph
            
            story.append(Paragraph(paragraph_text, normal_style))
            story.append(Spacer(1, 6))
        
        i += 1
    
    # Build the PDF
    doc.build(story)

def main():
    # Define directories
    input_dir = "datasets/scientists_bios"
    output_dir = os.path.join(input_dir, "PDFs")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all text files
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    # Convert each file
    for txt_file in txt_files:
        input_path = os.path.join(input_dir, txt_file)
        output_path = os.path.join(output_dir, txt_file.replace('.txt', '.pdf'))
        
        print(f"Converting {txt_file} to PDF...")
        try:
            convert_text_to_pdf(input_path, output_path)
            print(f"Saved as {output_path}")
        except Exception as e:
            print(f"Error converting {txt_file}: {e}")
    
    print(f"Conversion complete. {len(txt_files)} files processed.")

if __name__ == "__main__":
    main()