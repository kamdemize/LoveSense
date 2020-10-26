using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;

namespace LoveSense.Domaine
{
    public class Message
    {
        [Required(ErrorMessage ="Text to verify is required")]
        [StringLength(500, MinimumLength =100, ErrorMessage = "The text must have a minimum of 100 characters and a maximum of 500")]
        public string Text { get; set; }
    }
}
