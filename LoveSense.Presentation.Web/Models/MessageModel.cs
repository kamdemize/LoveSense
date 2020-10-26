using System.ComponentModel.DataAnnotations;

namespace LoveSense.Presentation.Web.Models
{
    public class MessageModel
    {
        [Required(ErrorMessage ="Text to verify is required")]
        [StringLength(500, MinimumLength =100, ErrorMessage = "The text must have a minimum of 100 characters and a maximum of 500")]
        public string Text { get; set; }

        public bool IsInVerify { get; set; }
        public string DescriptionResponse { get; set; }
    }
}
