using System.Collections.Generic;
using UnityEngine;

namespace Env5
{
    public class CollisionDetector : MonoBehaviour
    {
        // Start is called before the first frame update
        private HashSet<string> touchedObjects = new();

        public bool Touching(GameObject gameObject)
        {
            return Touching(gameObject.tag);
        }
        public bool Touching(string tag)
        {
            return touchedObjects.Contains(tag);
        }

        void OnCollisionEnter(Collision collision)
        {
            touchedObjects.Add(collision.gameObject.tag);
        }
        void OnCollisionStay(Collision collision)
        {
            touchedObjects.Add(collision.gameObject.tag);
        }
        void OnCollisionExit(Collision collision)
        {
            touchedObjects.Remove(collision.gameObject.tag);
        }
    }
}
