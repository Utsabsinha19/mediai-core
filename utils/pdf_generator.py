from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os
import base64
from io import BytesIO
from PIL import Image as PILImage

class PDFGenerator:
    def __init__(self, output_dir="backend/static/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5f7a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c7da0'),
            spaceBefore=20,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='Diagnosis',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#27ae60'),
            alignment=TA_CENTER,
            spaceAfter=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='DiagnosisAbnormal',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#e74c3c'),
            alignment=TA_CENTER,
            spaceAfter=20
        ))
    
    async def generate_medical_report(self, report_data: dict, user_data: dict) -> str:
        """Generate medical report PDF"""
        filename = f"report_{report_data['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        
        # 1. Header
        story.append(Paragraph("🏥 AI HEALTHCARE PLATFORM", self.styles['CustomTitle']))
        story.append(Paragraph("Medical Imaging Analysis Report", self.styles['SectionHeader']))
        story.append(Spacer(1, 20))
        
        # 2. Patient Information
        story.append(Paragraph("Patient Information", self.styles['SectionHeader']))
        patient_info = [
            ["Report ID:", report_data['id']],
            ["Patient Name:", user_data.get('full_name', user_data['username'])],
            ["Patient ID:", user_data['id']],
            ["Report Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Image Name:", report_data['image_name']]
        ]
        
        patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c7da0')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # 3. AI Prediction Results
        story.append(Paragraph("AI Analysis Results", self.styles['SectionHeader']))
        
        # Final diagnosis with color coding
        is_normal = report_data['prediction'].lower() == 'normal'
        diagnosis_style = 'Diagnosis' if is_normal else 'DiagnosisAbnormal'
        story.append(Paragraph(f"<b>Final Diagnosis: {report_data['prediction']}</b>", self.styles[diagnosis_style]))
        
        # Confidence score
        story.append(Paragraph(f"<b>Confidence Score:</b> {report_data['confidence']:.1%}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Best Model:</b> {report_data['best_model'].upper()}", self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        # 4. Model Comparison Table
        story.append(Paragraph("Model Performance Comparison", self.styles['SectionHeader']))
        
        comparison_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confidence']]
        for model_name, metrics in report_data['model_metrics'].items():
            comparison_data.append([
                model_name.upper(),
                f"{metrics.get('accuracy', 0):.1%}",
                f"{metrics.get('precision', 0):.1%}",
                f"{metrics.get('recall', 0):.1%}",
                f"{metrics.get('f1_score', 0):.1%}",
                f"{metrics.get('confidence', 0):.1%}"
            ])
        
        comparison_table = Table(comparison_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c7da0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(comparison_table)
        story.append(Spacer(1, 20))
        
        # 5. SHAP Explanation
        if report_data.get('shap_plot_base64'):
            story.append(Paragraph("AI Explainability (SHAP Analysis)", self.styles['SectionHeader']))
            
            # Decode base64 and add image
            image_data = base64.b64decode(report_data['shap_plot_base64'])
            img = PILImage.open(BytesIO(image_data))
            
            # Save temporary image
            temp_img_path = os.path.join(self.output_dir, f"temp_shap_{report_data['id']}.png")
            img.save(temp_img_path)
            
            # Add to PDF
            shap_image = Image(temp_img_path, width=5*inch, height=4*inch)
            story.append(shap_image)
            story.append(Spacer(1, 10))
            
            # Cleanup temp file
            os.remove(temp_img_path)
        
        # 6. Medical Recommendations
        story.append(Paragraph("Medical Recommendations", self.styles['SectionHeader']))
        recommendations = self._get_recommendations(report_data['prediction'])
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 7. Disclaimer
        story.append(Paragraph("Disclaimer", self.styles['SectionHeader']))
        disclaimer_text = """
        This report is generated by an AI-powered diagnostic system. The predictions and recommendations 
        are for informational purposes only and should not be considered as professional medical advice. 
        Always consult with a qualified healthcare provider for medical decisions. The AI system is a 
        supportive tool and not a replacement for clinical judgment.
        """
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def _get_recommendations(self, diagnosis: str) -> list:
        """Get medical recommendations based on diagnosis"""
        recommendations = {
            'Normal': [
                "Continue regular health check-ups",
                "Maintain healthy lifestyle habits",
                "Report any unusual symptoms to your doctor",
                "Follow up as recommended by your physician"
            ],
            'Glioma': [
                "IMMEDIATE CONSULTATION with a neurosurgeon recommended",
                "MRI with contrast for detailed evaluation",
                "Consider biopsy for definitive diagnosis",
                "Discuss treatment options: surgery, radiation, chemotherapy",
                "Neurological evaluation for symptom management"
            ],
            'Meningioma': [
                "Consult with a neurologist or neurosurgeon",
                "Regular MRI monitoring to track growth",
                "Consider surgical options if symptomatic",
                "Radiation therapy may be an option for inoperable cases",
                "Monitor for neurological symptoms"
            ],
            'Pituitary': [
                "Endocrinologist consultation recommended",
                "Hormone level testing (prolactin, GH, ACTH, TSH)",
                "MRI with contrast for detailed imaging",
                "Consider medication management for hormone-secreting tumors",
                "Surgical consultation for larger tumors"
            ]
        }
        return recommendations.get(diagnosis, recommendations['Normal'])